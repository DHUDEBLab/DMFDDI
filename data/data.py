from argparse import Namespace
import random
from typing import Callable, List, Union, Tuple

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from rdkit import Chem

from .scaler import StandardScaler
from .mol_tree import MolTree, MolTreeNode
from features import get_features_generator
import copy
import random
import pickle
from mol2vec.features import mol2alt_sentence

def load_pickle(filepath):
    obj = pickle.load(open(filepath, 'rb'))
    return obj


smiles2moltree = None

# MoleculeDatapoint 包含单个分子及其相关的特征和目标。
class MoleculeDatapoint:
    """MoleculeDatapoint 包含单个分子及其相关的特征和目标。"""

    smiles2moltree = None

    def __init__(self,
                 line: List[str],
                 args: Namespace = None,
                 features: np.ndarray = None,
                 use_compound_names: bool = False,
                 label_index=-1):
        """
         初始化包含单个分子的 MoleculeDatapoint。
         :param line：通过逗号分隔数据 CSV 文件中的一行生成的字符串列表。
         :param args: 参数。
         :param features：包含附加功能的 numpy 数组（例如 Morgan 指纹）。
         :param use_compound_names：数据 CSV 是否在每行包含化合物名称。
        """
        if args is not None:
            self.features_generator = args.features_generator
            self.args = args
        else:
            self.features_generator = self.args = None

        if features is not None and self.features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        self.features = features

        if use_compound_names:
            self.compound_name = line[0]  # str
            line = line[1:]
        else:
            self.compound_name = None

        self.smiles = line[0]  # str
        self.mol = Chem.MolFromSmiles(self.smiles)

        # 如果给定生成器，则生成附加功能
        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if self.mol is not None and self.mol.GetNumHeavyAtoms() > 0:
                    self.features.extend(features_generator(self.mol))

            self.features = np.array(self.features)

        # Fix nans in features修复功能中的 nans
        if self.features is not None:
            replace_token = 0
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Create targets创建目标
        self.targets = [float(x) if x != '' else None for x in line[1:]]

        # 用于连接树预测
        self.mol_tree = self.get_mol_tree() if self.args.jt else None

    def set_features(self, features: np.ndarray):
        """
          设置分子的特征。
         :param features：分子的一维 numpy 特征数组。
        """
        self.features = features

    def num_tasks(self) -> int:
        """
         返回预测任务的数量。
         :return: 任务数。
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
         设置分子的目标。
         :param 目标：包含目标的浮点数列表。
        """
        self.targets = targets

    def get_mol_tree(self) -> MolTree:
        assert self.args.jt == True
        global smiles2moltree
        if smiles2moltree is None:
            smiles2moltree = load_pickle(self.args.smiles2moltree_file)
        return smiles2moltree[self.smiles]

# MoleculeDataset 包含分子列表及其相关特征和目标。
class MoleculeDataset(Dataset):
    """MoleculeDataset 包含分子列表及其相关特征和目标。"""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
         初始化一个 MoleculeDataset，它包含一个 MoleculeDatapoints 列表（即分子列表）。
         :param data: 分子数据点列表。
        """
        self.data = data
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None

    def compound_names(self) -> List[str]:
        """
        返回与分子关联的化合物名称（如果存在）。
         :return: 化合物名称列表或 None 如果数据集不包含化合物名称。
        """
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def smiles(self) -> List[str]:
        """
        返回与分子关联的微笑字符串。
         :return: 微笑字符串列表。
        """
        return [d.smiles for d in self.data]
    
    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self.data]

    # 添加 jt 图模型
    def mol_trees(self) -> List[MolTree]:
        assert self.args.jt == True
        return [d.mol_tree for d in self.data if d.mol_tree.has_root()]

    def features(self) -> List[np.ndarray]:
        """
         返回与每个分子相关的特征（如果存在）。
         :return: 包含每个分子特征的一维 numpy 数组列表，如果没有特征，则为 None。
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        return [d.features for d in self.data]

    def targets(self) -> List[List[float]]:
        """
        返回与每个分子相关的目标
         :return: 包含目标的浮点数列表。
        """
        # add for jt graph model
        if not self.args.jt:
            return [d.targets for d in self.data]
        else:
            return [d.targets for d in self.data if d.mol_tree.has_root()]

    def num_tasks(self) -> int:
        """
        返回预测任务的数量。
         :return: 任务数。
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        """
        返回与每个分子关联的特征数组的大小。
         :return: 特征的大小。
        """
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None

    def shuffle(self, seed: int = None):
        """
         打乱数据集。
         :param 种子：可选的随机种子。
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        使用 StandardScaler（减去均值，除以标准差）对数据集的特征进行归一化。

         如果提供了缩放器，则使用该缩放器来执行归一化。 否则适合缩放器
         数据集中的特征，然后执行归一化。

         :param scaler: 一个合适的 StandardScaler。 提供时使用。 否则，StandardScaler 适合
         这个数据集然后被使用。
         :param replace_nan_token: 用什么替换 nans。
         :return: 一个合适的 StandardScaler。 如果提供了缩放器，则这是相同的缩放器。 否则，这是
         适合此数据集的缩放器。
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler
    
    def set_targets(self, targets: List[List[float]]):
        """
         为数据集中的每个分子设置目标。 假设目标与数据点对齐。

         :param 目标：包含每个分子的目标的浮点列表列表。 这必须是与基础数据集的长度相同。
        """
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
         使用提供的键对数据集进行排序。

         :param key: 一个用于确定排序顺序的 MoleculeDatapoint 函数。
        """
        self.data.sort(key=key)

    def __len__(self) -> int:
        """
            返回数据集的长度（即分子数）。

         :return: 数据集的长度。
        """
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint], dict]:
        """
            通过索引或切片获取一个或多个 MoleculeDatapoints。

         :param item: 索引 (int) 或切片对象。
         :return: 如果提供了 int，则为 MoleculeDatapoint；如果提供了切片，则为 MoleculeDatapoints 列表。
        """
        return self.data[item]

class DDIDatapoint:
    """MoleculeDatapoint 包含单个 ddi 及其关联的特征和目标。"""

    def __init__(self,
                 dictionary: dict,
                 args: Namespace = None,
                 features_1: np.ndarray = None,
                 features_2: np.ndarray = None,
                 ):
        """
        初始化包含单个分子的 MoleculeDatapoint。

         :param line：通过逗号分隔数据 CSV 文件中的一行生成的字符串列表。
         :param args: 参数。
         :param features：包含附加功能的 numpy 数组（例如 Morgan 指纹）。
         :param use_compound_names：数据 CSV 是否在每行包含化合物名称。
        """
        if args is not None:
            self.features_generator = args.features_generator
            self.args = args
        else:
            self.features_generator = self.args = None

        if features_1 is not None and self.features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')
        if features_2 is not None and self.features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        self.features_1 = features_1
        self.features_2 = features_2

        self.smiles_1 = dictionary['smiles_1']
        self.smiles_2 = dictionary['smiles_2']

        self.mol_1 = Chem.MolFromSmiles(self.smiles_1)
        self.mol_2 = Chem.MolFromSmiles(self.smiles_2)

        # 如果给定生成器，则生成附加功能
        if self.features_generator is not None:
            self.features_1 = []
            self.features_2 = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if self.mol_1 is not None and self.mol_1.GetNumHeavyAtoms() > 0:
                    self.features_1.extend(features_generator(self.smiles_1))
                if self.mol_2 is not None and self.mol_2.GetNumHeavyAtoms() > 0:
                    self.features_2.extend(features_generator(self.smiles_2))

            self.features_1 = np.array(self.features_1)
            self.features_2 = np.array(self.features_2)

        # Fix nans in features
        if self.features_1 is not None:
            replace_token = 0
            self.features_1 = np.where(np.isnan(self.features_1), replace_token, self.features_1)
        # Fix nans in features
        if self.features_2 is not None:
            replace_token = 0
            self.features_2 = np.where(np.isnan(self.features_2), replace_token, self.features_2)

        # Create targets
        if args.num_labels is not None:
            target_scalar = int(dictionary['label'])
            self.targets = [target_scalar]
        else:
            target_scalar = int(dictionary['label'])
            self.targets = [target_scalar]

        # 用于连接树预测
        self.mol_tree_pair = self.get_mol_tree_pair() if self.args.jt else None

    def set_features(self, features_1: np.ndarray, features_2: np.ndarray):
        """
       设置分子的特征。

         :param features：分子的一维 numpy 特征数组。
        """
        self.features_1 = features_1
        self.features_2 = features_2

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        # if self.args.dataset_type == 'multilabel':
        #     return self.args.num_labels
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def mirror_features(self):
        temp = self.features_1
        self.features_1 = self.features_2
        self.features_2 = temp

    def mirror_smiles_pair(self):
        temp = self.smiles_1
        self.smiles_1 = self.smiles_2
        self.smiles_2 = temp

    def mirror_mol_pair(self):
        temp = self.mol_1
        self.mol_1 = self.mol_2
        self.mol_2 = temp

    # for jt graph model
    def mirror_mol_tree_pair(self):
        temp = self.mol_tree_pair[0]
        self.mol_tree_pair[0] = self.mol_tree_pair[1]
        self.mol_tree_pair[1] = temp

    def mirror(self):
        ano_datapoint = copy.deepcopy(self)
        ano_datapoint.mirror_features()
        ano_datapoint.mirror_smiles_pair()
        ano_datapoint.mirror_mol_pair()
        # for jt graph mdoel
        if self.args.jt:
            ano_datapoint.mirror_mol_tree_pair()
        return ano_datapoint

    # for jt graph model
    def get_mol_tree_pair(self) -> Tuple[MolTree, MolTree]:
        assert self.args.jt == True
        global smiles2moltree
        if smiles2moltree is None:
            smiles2moltree = load_pickle(self.args.smiles2moltree_file)
        return (smiles2moltree[self.smiles_1], smiles2moltree[self.smiles_2])


class DDIDataset(Dataset):
    """MoleculeDataset 包含分子列表及其相关特征和目标。"""

    def __init__(self, data: List[DDIDatapoint]):
        """
        初始化一个 MoleculeDataset，它包含一个 MoleculeDatapoints 列表（即分子列表）。

         :param data: 分子数据点列表。
        """
        self.data = data
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None

    def augment(self):
        data_mirror = [d.mirror() for d in self.data]
        data_aug = copy.deepcopy(self.data)
        data_aug.extend(data_mirror)
        random.shuffle(data_aug)
        assert len(data_aug) == 2 * len(self.data)
        return DDIDataset(data_aug)

    def mirror(self):
        data_mirror = [d.mirror() for d in self.data]
        return DDIDataset(data_mirror)

    def smiles_pairs(self):
        """
        返回与分子关联的微笑字符串。

         :return: 微笑字符串列表。
        """
        return [(d.smiles_1, d.smiles_2) for d in self.data]

    def mol_pairs(self):
        """
        返回与分子关联的 RDKit 分子。

         :return: RDKit Mols 列表。
        """
        return [(d.mol_1, d.mol_2) for d in self.data]

    # add for jt graph model
    def mol_tree_pairs(self) -> List[Tuple[MolTree, MolTree]]:
        assert self.args.jt == True
        return [d.mol_tree_pair for d in self.data
                if d.mol_tree_pair[0].has_root() \
                and d.mol_tree_pair[1].has_root()
                ]

    def features_pairs(self):
        """
        返回与每个分子相关的特征（如果存在）。

         :return: 包含每个分子特征的一维 numpy 数组列表，如果没有特征，则为 None。
        """
        if len(self.data) == 0 or self.data[0].features_1 is None or self.data[0].features_2 is None:
            return None

        return [(d.features_1, d.features_2) for d in self.data]

    def targets(self) -> List[List[int]]:
        """
         返回与每个分子相关的目标。
         :return: 包含目标的浮点数列表。
        """
        if not self.args.jt:
            return [d.targets for d in self.data]
        # for jt graph model
        else:
            return [d.targets for d in self.data if d.mol_tree_pair[0].has_root() and d.mol_tree_pair[1].has_root()]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        """
        返回与每个分子关联的特征数组的大小。

         :return: 特征的大小。
        """
        return len(self.data[0].features_1) if len(self.data) > 0 and self.data[0].features_1 is not None else None

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
         使用 StandardScaler（减去均值，除以标准差）对数据集的特征进行归一化。

         如果提供了缩放器，则使用该缩放器来执行归一化。 否则适合缩放器
         数据集中的特征，然后执行归一化。

         :param scaler: 一个合适的 StandardScaler。 提供时使用。 否则，StandardScaler 适合
         这个数据集然后被使用。
         :param replace_nan_token: 用什么替换 nans。
         :return: 一个合适的 StandardScaler。 如果提供了缩放器，则这是相同的缩放器。 否则，这是
         适合此数据集的缩放器。
        """
        if len(self.data) == 0 or self.data[0].features_1 is None or self.data[0].features_2 is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features_1 = np.vstack([d.features_1 for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features_1)

            features_2 = np.vstack([d.features_2 for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features_2)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features_1.reshape(1, -1))[0],
                           self.scaler.transform(d.features_2.reshape(1, -1)[0]))

        return self.scaler

    def set_targets(self, targets: List[List[int]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each molecule. This must be the
        same length as the underlying dataset.
        """
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        self.data.sort(key=key)

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        """
        通过索引或切片获取一个或多个 MoleculeDatapoints。

         :param item: 索引 (int) 或切片对象。
         :return: 如果提供了 int，则为 MoleculeDatapoint；如果提供了切片，则为 MoleculeDatapoints 列表。
        """
        return self.data[item]


# Memoization
SMILES_TO_SENTENCE = {}

# 添加选项“半径”
def mol2sentence(smiles_batch: List[str], vocab, args: Namespace) -> List[dict]:
    output_list = []
    for smiles in smiles_batch:
        if smiles in SMILES_TO_SENTENCE:
            sentence = SMILES_TO_SENTENCE[smiles]
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                sentence = mol2alt_sentence(mol, radius=args.radius)
                SMILES_TO_SENTENCE[smiles] = sentence
            else:
                continue
        # convert to ids
        sentence = [vocab.stoi.get(token, vocab.unk_index) for i, token in enumerate(sentence)]
        sentence = [vocab.sos_index] + sentence + [vocab.eos_index]
        segment_label = ([1 for _ in range(len(sentence))])[:args.seq_len]

        input = sentence[:args.seq_len]
        padding = [vocab.pad_index for _ in range(args.seq_len - len(input))]
        input.extend(padding)
        segment_label.extend(padding)

        output = {
            'input': input,
            'segment_label': segment_label
        }

        output = {key: torch.tensor(value) for key, value in output.items()}
        output_list.append(output)

    return output_list



