import csv
import os
import pickle
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

def save_features(path: str, features: List[np.ndarray]):
    """
    Saves features to a compressed .npz file with array name "features".

    :param path: Path to a .npz file where the features will be saved.
    :param features: A list of 1D numpy arrays containing the features for molecules.
    """
    # 将特征保存到一个压缩的.npz文件，数组名称为“features”。
    # :param path: 保存特性的.npz文件的路径。
    # :param features：包含分子特征的一维numpy数组列表。
    np.savez_compressed(path, features=features)


def load_features(path: str) -> np.ndarray:
    """
    加载以各种格式保存的特征。

     支持的格式：
     - .npz 压缩（假设特征以名称“特征”保存）
     - .npz（假设特征以名称“特征”保存）
     - .npy
     - .csv/.txt（假设逗号分隔的特征带有标题和每个分子一行）
     - .pkl/.pckl/.pickle 包含一个稀疏的 numpy 数组（待办事项：一旦我们不再依赖它，就删除这个选项）

     所有格式都假定代码中其他地方加载的 SMILES 字符串在相同的
     按此处加载的功能排序。

     :param path: 包含特征的文件的路径。
     :return: 包含特征的大小为 (num_molecules, features_size) 的 2D numpy 数组。
    """
    extension = os.path.splitext(path)[1]

    if extension == '.npz':
        features = np.load(path)['features']
    elif extension == '.npy':
        features = np.load(path)
    elif extension in ['.csv', '.txt']:
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            features = np.array([[float(value) for value in row] for row in reader])
    elif extension in ['.pkl', '.pckl', '.pickle']:
        with open(path, 'rb') as f:
            features = np.array([np.squeeze(np.array(feat.todense())) for feat in pickle.load(f)])
    else:
        raise ValueError(f'Features path extension {extension} not supported.')

    return features


class MolFeatureExtractionError(Exception):
    pass


def type_check_num_atoms(mol, num_max_atoms=-1):
    num_atoms = mol.GetNumAtoms()
    if num_max_atoms >= 0 and num_atoms > num_max_atoms:
        raise MolFeatureExtractionError(
            'Number of atoms in mol {} exceeds num_max_atoms {}'
            .format(num_atoms, num_max_atoms))

# 构建原子序数列表
def construct_atomic_number_array(mol, out_size=-1):
    atom_list = [a.GetAtomicNum() for a in  mol.GetAtoms()] #返回原子序数。
    n_atoms = len(atom_list)

    if out_size < 0:
        return np.array(atom_list, dtype=np.int32)
    elif out_size >= n_atoms:
        atom_array = np.zeros(out_size, dtype=np.int32)
        atom_array[:n_atoms] = atom_list
        return atom_array
    else:
        raise ValueError('`out_size` (={}) must be negative or '
                         'larger than or equal to the number '
                         'of atoms in the input molecules (={})'
                         '.'.format(out_size, n_atoms))

# 构建adj序数列表
def construct_adj_matrix(mol, out_size=-1, self_connections=True):
    adj = rdmolops.GetAdjacencyMatrix(mol)
    s0, s1 = adj.shape
    if s0 != s1:
        raise ValueError('The adjacent matrix of the input molecule'
                         'has an invalid shape: ({}, {}). '
                         'It must be square.'.format(s0, s1))

    if self_connections:
        adj += np.eye(s0)
    if out_size < 0:
        adj_array = adj.astype(np.float32)
    elif out_size >= 0:
        adj_array = np.zeros((out_size, out_size), dtype=np.float32)
        adj_array[:s0, :s1] = adj
    else:
        raise ValueError(
            '`out_size` (={}) must be negative or larger than or equal to the '
            'number of atoms in the input molecules (={}).'
            .format(out_size, s0))

    return adj_array


def construct_discrete_edge_matrix(mol, out_size=-1):
    if mol is None:
        raise MolFeatureExtractionError('mol is None')
    N = mol.GetNumAtoms()

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise ValueError(
            'out_size {} is smaller than number of atoms in mol {}'
            .format(out_size, N))

    adjs = np.zeros((4, size, size), dtype=np.float32)

    bond_type_to_channel = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3,
    }

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        ch = bond_type_to_channel[bond_type]
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjs[ch, i, j] = 1.0
        adjs[ch, j, i] = 1.0

    return adjs