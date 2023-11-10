from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.sparse import block_diag

from . import mask, array


def dropna_block_diag(genotype, covariates):
    As = []
    for i in range(genotype.shape[1]):
        A = mask.dropnan(array.concat((genotype[:, i:i+1], covariates)))
        As.append(A)
    return block_diag(As)


class AbstractBlockDiagonalMatrix(ABC):

    def __init__(self):
        pass

    @property
    def ndim(self):
        return 2

    @property
    @abstractmethod
    def nblocks(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def blockshapes(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, key):
        raise NotImplementedError()


class BlockDiagonalMatrixIterator:

    def __init__(self, bd: AbstractBlockDiagonalMatrix) -> None:
        self.__bd = bd
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.__bd.nblocks:
            block = self.__bd[self.index]
            self.index += 1
            return block
        else:
            raise StopIteration


class BlockDiagonalMatrix(AbstractBlockDiagonalMatrix):
    """Block diagonal matrix which stores dense numpy matrices separately."""

    def __init__(self, blocks: List[np.ndarray]) -> None:
        super().__init__()
        self.__blocks = blocks

    @property
    def nblocks(self):
        return len(self.__blocks)

    @property
    def blocks(self):
        return self.__blocks

    @property
    def blockshapes(self):
        return [blk.shape for blk in self.blocks]

    def blockshape(self, i: int):
        return self.blocks[i].shape

    def append(self, x: np.ndarray):
        return self.__blocks.append(x)

    def __iter__(self):
        return BlockDiagonalMatrixIterator(self)

    def __getitem__(self, key):
        if key < self.nblocks:
            return self.__blocks[key]
        else:
            raise IndexError
