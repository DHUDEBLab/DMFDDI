from typing import Union, Tuple, List, Dict

from .decoder import InnerProductDecoder

from models.ggnn import AGRUNN

from data.mol_tree import Vocab
from nn_utils import get_activation_function, initialize_weights


import torch
import numpy as np
import torch.nn as nn
from argparse import Namespace
from typing import Union, List
from features.featurization import BatchMolGraph, mol2graph

from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .losses_info import local_global_drug_loss_
from .model_info import PriorDiscriminator, FF_local, FF_global

class InnerProductDecoder(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs_row = inputs
        inputs_col = inputs.transpose(0, 1)
        inputs_row = self.dropout(inputs_row)
        inputs_col = self.dropout(inputs_col)
        rec = torch.mm(inputs_row, inputs_col)
        outputs = self.act(rec)
        return outputs

class MffInfomax(nn.Module):
    def __init__(self, args: Namespace, gamma=.1):
        super(MffInfomax, self).__init__()
        self.args = args
        self.gamma = gamma
        self.prior = args.prior
        self.features_dim = args.hidden_size
        self.embedding_dim = args.gcn_hidden3
        self.local_d = FF_local(args, self.features_dim)
        self.global_d = FF_global(args, self.embedding_dim)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

    def forward(self, embeddings, features, adj_tensor, num_drugs):

        g_enc = self.global_d(embeddings)
        l_enc = self.local_d(features)
        measure = 'JSD'
        local_global_loss = local_global_drug_loss_(self.args, l_enc, g_enc, adj_tensor, num_drugs, measure)
        eps = 1e-5
        if self.prior:
            prior = torch.rand_like(embeddings)
            term_a = torch.log(self.prior_d(prior) + eps).mean()
            term_b = torch.log(1.0 - self.prior_d(embeddings) + eps).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR

class GGNNEncoder(nn.Module):
    def __init__(self, args: Namespace, atom_fdim: int, mol_fdim: int):
        super(GGNNEncoder, self).__init__()
        # predefined for GGNNEncoder
        self.num_atom_types = 117
        self.num_edge_types = 4
        num_layers = args.depth
        weight_tying = False
        concat_hidden = False
        num_message_layers = 1 if weight_tying else num_layers
        num_readout_layers = num_layers if concat_hidden else 1
        self.embed = nn.Embedding(self.num_atom_types, atom_fdim)
        self.message_layers = nn.ModuleList([nn.Linear(atom_fdim, self.num_edge_types * atom_fdim) for _ in range(num_message_layers)])
        self.update_layer = nn.GRUCell(atom_fdim, atom_fdim)
        # self.update_layer = nn.RNNCell(atom_fdim, atom_fdim)
        self.i_layers = nn.ModuleList([nn.Linear(2 * atom_fdim, mol_fdim) for _ in range(num_readout_layers)])
        self.j_layers = nn.ModuleList([nn.Linear(atom_fdim, mol_fdim) for _ in range(num_readout_layers)])

        self.args = args
        self.atom_fdim = atom_fdim
        self.mol_fdim = mol_fdim
        self.num_layers = num_layers
        self.atom_fdim = atom_fdim
        self.weight_tying = weight_tying
        self.concat_hidden = concat_hidden
        self.use_input_features = args.use_input_features

    def forward(self, mol_graph: BatchMolGraph, features_batch: List[np.ndarray] = None):
        args = self.args
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        atom_array, adj = mol_graph.get_ggnn_features()

        if args.cuda:
            atom_array = atom_array.cuda()
            adj = adj.cuda()

        # self.update_layer.reset_parameters()

        # embedding layer
        h = self.embed(atom_array)
        h0 = h.clone()

        # message passing
        for step in range(self.num_layers):
            h = self.update(h, adj, step)

        g = self.readout(h, h0, step=0)
        return g

    def update(self, h: torch.Tensor, adj: torch.Tensor, step: int = 0):
        mb, atom, _ = h.shape
        message_layer_index = 0 if self.weight_tying else step
        # h: (mb, atom, atom_fdim) -> (mb, atom, num_edge_types * atom_fdim)
        # m: (mb, atom, num_edge_types, atom_fdim)
        m = self.message_layers[message_layer_index](h).view(
            mb, atom, self.num_edge_types, self.atom_fdim
        )
        # m: (mb, num_edge_types, atom, atom_fdim)
        m = m.permute(0, 2, 1, 3)
        m = m.contiguous()
        # m: (mb * num_edge_types, atom, atom_fdim)
        m = m.view(mb * self.num_edge_types, atom, self.atom_fdim)
        # adj: (mb * num_edge_types, atom, atom)
        adj = adj.view(mb * self.num_edge_types, atom, atom)
        # m: (mb * num_edge_types, atom, atom_fdim)
        m = torch.bmm(adj, m)
        # m: (mb, num_edge_types, atom, atom_fdim)
        m = m.view(mb, self.num_edge_types, atom, self.atom_fdim)
        # m: (mb, atom, atom_fdim)
        m = torch.sum(m, dim=1)

        # update via GRUCell
        # m: (mb * atom, atom_fdim)
        m = m.view(mb * atom, -1)
        # h: (mb * atom, atom_fdim)
        h = h.view(mb * atom, -1)
        # out_h: (mb * atom, atom_fdim)
        out_h = self.update_layer(m, h if step > 0 else None)
        out_h = out_h.view(mb, atom, self.atom_fdim)
        return out_h

    def readout(self, h, h0, step):
        index = step if self.concat_hidden else 0
        a=torch.mean(torch.sigmoid(self.i_layers[index](torch.cat([h, h0], dim=2))) * self.j_layers[index](h), dim=1)
        b=torch.sum(torch.sigmoid(self.i_layers[index](torch.cat([h, h0], dim=2))) * self.j_layers[index](h), dim=1)
        return (a+b)/2 #sum mean

class AGRUNN(nn.Module):
    def __init__(self, args: Namespace, graph_input: bool = False):
        super(AGRUNN, self).__init__()
        self.args = args
        self.graph_input = graph_input
        self.encoder = GGNNEncoder(args, atom_fdim=args.hidden_size, mol_fdim=args.hidden_size)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> Union[torch.FloatTensor, torch.Tensor]:

        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)
        output = self.encoder.forward(batch, features_batch)

        return output

class SparseGraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(SparseGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, input: torch.sparse.FloatTensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNEncoder(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.1, bias: bool = False,
                 sparse: bool = True):
        super(GCNEncoder, self).__init__()
        self.input_dim = num_features
        self.features_nonzero = features_nonzero if sparse else None
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.sparse = sparse
        self.gc1 = SparseGraphConvolution(in_features=num_features, out_features=args.hidden1, bias=bias)
        self.gc2 = SparseGraphConvolution(in_features=args.hidden1, out_features=args.hidden2, bias=bias)

    def forward(self, features: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        if not self.sparse:
            features = self.dropout(features)
        hidden1 = F.relu(self.gc1(features, adj))
        hidden1 = self.dropout(hidden1)
        embeddings = self.gc2(hidden1, adj)
        return embeddings

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class GCNEncoderWithFeatures(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.1, bias: bool = False,
                 sparse: bool = True):
        super(GCNEncoderWithFeatures, self).__init__()
        self.input_dim = num_features #300
        self.dropout = nn.Dropout(dropout) #0.3
        self.bias = bias #True
        self.sparse = sparse
        self.gc_input = SparseGraphConvolution(in_features=num_features, out_features=args.gcn_hidden1, bias=bias) #300 300 true
        self.gc_hidden1 = SparseGraphConvolution(in_features=args.gcn_hidden1, out_features=args.gcn_hidden2, bias=bias) #300 281 true
        self.gc_hidden2 = SparseGraphConvolution(in_features=args.gcn_hidden2, out_features=args.gcn_hidden3, bias=bias)#281 264 true
        self.trans_h = nn.Linear(args.gcn_hidden1 + num_features, args.gcn_hidden1, bias=True) #600 300
        self.trans_h1 = nn.Linear(args.gcn_hidden2 + num_features, args.gcn_hidden2, bias=True) #581 281
        self.trans_h2 = nn.Linear(args.gcn_hidden3 + num_features, args.gcn_hidden3, bias=True)

    def forward(self, features: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        if not self.sparse:
            features = self.dropout(features)
        hidden1 = F.relu(self.trans_h(torch.cat([self.gc_input(features, adj), features], dim=1)))
        hidden1 = self.dropout(hidden1)
        hidden2 = F.relu(self.trans_h1(torch.cat([self.gc_hidden1(hidden1, adj), features], dim=1)))
        hidden2 = self.dropout(hidden2)
        embeddings = F.relu(self.trans_h2(torch.cat([self.gc_hidden2(hidden2, adj), features], dim=1)))
        return embeddings

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class MFGCN(nn.Module):
    def __init__(self, args: Namespace, num_features: int, features_nonzero: int,
                 dropout: float = 0.3, bias: bool = False,
                 sparse: bool = True):
        super(MFGCN, self).__init__()
        self.num_features = num_features #300
        self.features_nonzero = features_nonzero #0
        self.dropout = dropout #0.3
        self.bias = bias #true
        self.sparse = sparse #false
        self.args = args
        self.g_encoder(args)
        self.n_encoder = self.nn_encoder(args)
        self.dec_local = InnerProductDecoder(args.hidden_size)
        self.dec_global = InnerProductDecoder(args.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fuu()
        self.create_ffn(args)
        self.attention = Attention(512)

    def g_encoder(self, args: Namespace, vocab: Vocab = None):
        self.encoder = AGRUNN(args)
        return self.encoder

    def nn_encoder(self, args: Namespace):
        return GCNEncoderWithFeatures(args, self.num_features + self.args.input_features_size,
                                          self.features_nonzero,
                                          dropout=self.dropout, bias=self.bias,
                                          sparse=self.sparse)

    def create_ffn(self, args: Namespace):
        dropout = nn.Dropout(args.dropout) #0.3
        activation = get_activation_function(args.activation) #RELU

        self.fusion_local = nn.Linear(args.hidden_size, args.ffn_hidden_size)
        self.fusion_global = nn.Linear(args.gcn_hidden3, args.ffn_hidden_size)
        ffn = []
        for _ in range(args.ffn_num_layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(args.ffn_hidden_size, args.drug_nums),
        ])
        # Create FFN model
        self.ffn = nn.Sequential(*ffn)
        self.dropout = dropout

    def fuu(self):
        self.MFF_model = MffInfomax(self.args)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                adj: torch.sparse.FloatTensor,
                adj_tensor,
                drug_nums,
                return_embeddings: bool = False) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if self.args.use_input_features:
            smiles_batch = batch[0]
            features_batch = batch[1]
        else:
            smiles_batch = batch
            features_batch = None

        feat_orig = self.encoder(smiles_batch, features_batch)
        feat = self.dropout(feat_orig)
        fused_feat = self.fusion_local(feat)
        output = self.ffn(fused_feat)
        outputs = self.sigmoid(output)
        outputs_l = outputs.view(-1)

        embeddings_orig = self.n_encoder(feat_orig, adj)
        feat_n = self.dropout(embeddings_orig)
        fused_feat_n = self.fusion_global(feat_n)
        output_n = self.ffn(fused_feat_n)
        outputs_ = self.sigmoid(output_n)
        outputs_n = outputs_.view(-1)

        local_embed = feat_orig

        MFF = self.MFF_model(embeddings_orig, local_embed, adj_tensor, drug_nums)

        # fembeddings = torch.stack([outputs, outputs_], dim=1)
        # emb_last, att_all_nterview = self.attention(fembeddings)
        #embeddings, local_embed
        if return_embeddings:
            return outputs_, embeddings_orig
        return outputs_n, feat_orig, embeddings_orig, outputs_l, outputs_n, MFF