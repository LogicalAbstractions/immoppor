import random
from typing import List, Dict

from torch.utils.data import Dataset


class IndexDataset(Dataset):

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.__dataset__ = dataset
        self.__indices__ = indices

    def __len__(self):
        return len(self.__indices__)

    def __getitem__(self, item: int):
        index = self.__indices__[item]
        return self.__dataset__[index]


def _generate_random_split_indices_(size: int, partitions: Dict[str, float], seed: int = 1234) -> Dict[str, List[int]]:
    values = [x for x in range(0, size)]
    random.Random(seed).shuffle(values)

    current_index = 0
    result = dict()

    for k, v in partitions.items():
        partition_size = int(float(size) * v)
        result[k] = values[current_index:current_index + partition_size]

        current_index += partition_size

    return result


def random_split(self: Dataset, split: float, seed: int = 1234) -> tuple[Dataset, Dataset]:
    partitions = {"a": split, "b": (1.0 - split)}
    indices = _generate_random_split_indices_(len(self), partitions, seed)

    return IndexDataset(self, indices["a"]), IndexDataset(self, indices["b"])


def take(self: Dataset, count: int) -> Dataset:
    actualCount = count if count < len(self) else len(self)
    indices = [x for x in range(actualCount)]
    return IndexDataset(self, indices)


