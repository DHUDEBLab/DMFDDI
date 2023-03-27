from collections import defaultdict
import logging
import random
from typing import Dict, List, Set, Tuple, Union

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np

from .data import MoleculeDataset


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    计算 SMILES 字符串的 Bemis-Murcko 脚手架。

     :param mol: 微笑字符串或 RDKit 分子。
     :param include_chirality: 是否包含手性。
     ：返回：
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    计算每个微笑字符串的脚手架并返回从脚手架到微笑集的映射。

     :param mols: 微笑字符串或 RDKit 分子的列表。
     :param use_indices: 是否映射到 all_smiles 中的微笑索引而不是映射到微笑字符串本身。 如果有重复的微笑，这是必要的。
     :return: 将每个独特的脚手架映射到所有具有该脚手架的微笑（或微笑索引）的字典。
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(data: MoleculeDataset,
                   sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                   balanced: bool = False,
                   seed: int = 0,
                   logger: logging.Logger = None) -> Tuple[MoleculeDataset,
                                                           MoleculeDataset,
                                                           MoleculeDataset]:
    """
    按支架拆分数据集，以便没有共享支架的分子处于同一拆分中。

     :param data: 一个分子数据集。
     :param 大小：长度为 3 的元组，其中包含数据的比例
     训练、验证和测试集。
     :param平衡：尝试平衡每个集合中脚手架的大小，而不是只将最小的放在测试集中。
     :param seed: 进行平衡分裂时用于洗牌的种子。
     :param logger: 一个记录器。
     :return: 包含数据的训练、验证和测试拆分的元组。
    """
    assert sum(sizes) == 1

    # Split
    train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # 从脚手架映射到数据中的索引
    scaffold_to_indices = scaffold_to_smiles(data.mols(), use_indices=True)

    if balanced:  # 将大于 val/test 大小一半的东西放入训练中，剩下的只是随机排序
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    if logger is not None:
        logger.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
                     f'train scaffolds = {train_scaffold_count:,} | '
                     f'val scaffolds = {val_scaffold_count:,} | '
                     f'test scaffolds = {test_scaffold_count:,}')
    
    log_scaffold_stats(data, index_sets, logger=logger)

    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)


def log_scaffold_stats(data: MoleculeDataset,
                       index_sets: List[Set[int]],
                       num_scaffolds: int = 10,
                       num_labels: int = 20,
                       logger: logging.Logger = None) -> List[Tuple[List[float], List[int]]]:
    """
    记录并返回有关分子支架中计数和平均目标值的统计信息。

     :param data: 一个分子数据集。
     :param index_sets：表示数据分割的索引集列表。
     :param num_scaffolds：显示统计信息的脚手架数量。
     :param num_labels: 显示统计信息的标签数量。
     :param logger: 一个记录器。
     :return: 元组列表，其中每个元组包含一个平均目标值列表
     跨第一个 num_labels 标签和非零值的数量列表
     第一个 num_scaffolds 脚手架，按脚手架频率降序排列。
     """
     # 打印一些关于脚手架的统计数据
    target_avgs = []
    counts = []
    for index_set in index_sets:
        data_set = [data[i] for i in index_set]
        targets = [d.targets for d in data_set]
        targets = np.array(targets, dtype=np.float)
        target_avgs.append(np.nanmean(targets, axis=0))
        counts.append(np.count_nonzero(~np.isnan(targets), axis=0))
    stats = [(target_avgs[i][:num_labels], counts[i][:num_labels]) for i in range(min(num_scaffolds, len(target_avgs)))]

    if logger is not None:
        logger.debug('Label averages per scaffold, in decreasing order of scaffold frequency,'
                     f'capped at {num_scaffolds} scaffolds and {num_labels} labels: {stats}')

    return stats
