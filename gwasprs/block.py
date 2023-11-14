from abc import ABC, abstractmethod
from typing import List

import numpy as np
from scipy.sparse import issparse
from scipy.linalg import block_diag
import jax

from . import mask, array


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
        checks = [isinstance(x, (np.ndarray, np.generic, jax.Array)) or issparse(x) for x in blocks]
        assert np.all(checks)
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
        assert isinstance(x, (np.ndarray, np.generic, jax.Array)) or issparse(x)
        return self.__blocks.append(x)

    def __iter__(self):
        return BlockDiagonalMatrixIterator(self)

    def __getitem__(self, key):
        if key < self.nblocks:
            return self.__blocks[key]
        else:
            raise IndexError

    def __matmul__(self, value):
        if isinstance(value, AbstractBlockDiagonalMatrix):
            return BlockDiagonalMatrix([x @ y for (x, y) in zip(self.blocks, value.blocks)])
        elif isinstance(value, (np.ndarray, np.generic, jax.Array)) and value.ndim == 1:
            rowidx = np.cumsum([0] + [shape[0] for shape in self.blockshapes])
            colidx = np.cumsum([0] + [shape[1] for shape in self.blockshapes])
            res = np.empty(rowidx[-1])
            for i in range(self.nblocks):
                res.view()[rowidx[i]:rowidx[i+1]] = self[i] @ value.view()[colidx[i]:colidx[i+1]]
            return res

    def toarray(self):
        return block_diag(*self.blocks)

    def diagonal(self):
        return np.concatenate([blk.diagonal() for blk in self.blocks])


def dropna_block_diag(genotype, covariates):
    As = BlockDiagonalMatrix([])
    for i in range(genotype.shape[1]):
        A = mask.dropnan(array.concat((genotype[:, i:i+1], covariates)))
        As.append(A)
    return As
