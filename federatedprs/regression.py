from abc import ABCMeta
from typing import Any

import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp
from .stats import gen_mvdot, mvdot, unnorm_autocovariance, unnorm_covariance


class LinearSolver(object, metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    
class InverseSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
        inv_X = jnp.linalg.inv(X)
        # beta = X^-1 y
        return mvdot(inv_X, y)
    
class CholeskySolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
        # L = Cholesky(X)
        L = jnp.linalg.cholesky(X)
        # solve Lz = y
        z = jsp.linalg.solve_triangular(L, y, lower=True)
        # solve Lt beta = z
        return jsp.linalg.solve_triangular(L, z, trans="T", lower=True)
    
class QRSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
        # Q, R = QR(X)
        Q, R = jnp.linalg.qr(X)
        # solve R beta = Qty
        return jsp.linalg.solve(R, unnorm_covariance(Q, y), lower=False)


class LinearRegression:
    """A class for federated linear regression

    Args:
        beta (np.ndarray[(1,), np.floating], optional): _description_. Defaults to None.
        XtX (np.ndarray[(1, 1), np.floating], optional): _description_. Defaults to None.
        Xty (np.ndarray[(1,), np.floating], optional): _description_. Defaults to None.
        is_inv (bool, optional): if provided XtX is inversed or not. Defaults to False.
    """
    
    def __init__(self, beta = None, XtX = None, Xty = None, is_inv = False) -> None:
        if beta is None:
            if XtX is None or Xty is None:
                raise ValueError("Must provide XtX and Xty, since beta is not provided.")
            
            if is_inv:
                inv_XtX = XtX
            else:
                inv_XtX = jnp.linalg.inv(XtX)
            
            beta = mvdot(XtX, Xty)
            
        self.__beta = beta
    
    def predict(self, X: np.ndarray[(1, 1), np.floating]):
        f = gen_mvdot(self.__beta)
        return f(X)

    @classmethod
    def fit(cls, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating], algo=CholeskySolver()):
        if isinstance(algo, QRSolver):
            beta = algo(X, y)
        else:
            beta = algo(unnorm_autocovariance(X), unnorm_covariance(X, y))
        return LinearRegression(beta = beta)
    
    def residual(self, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
        return y - self.predict(X)
    
    def sse(self, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
        res = self.residual(X, y)
        return jnp.vdot(res.T, res)
    