from abc import ABCMeta

import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp
from .stats import gen_mvdot, mvdot, unnorm_autocovariance, unnorm_covariance


class LRTrainer(object, metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    
class InverseLRTrainer(LRTrainer):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
        inv_XtX = jnp.linalg.inv(unnorm_autocovariance(X))
        # beta = (XtX)^-1 Xty
        beta = mvdot(inv_XtX, unnorm_covariance(X, y))
        return beta
    
class CholeskyLRTrainer(LRTrainer):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
        # L = Cholesky(XtX)
        L = jnp.linalg.cholesky(unnorm_autocovariance(X))
        # solve Lz = Xty
        z = jsp.linalg.solve_triangular(L, unnorm_covariance(X, y), lower=True)
        # solve Lt beta = z
        beta = jsp.linalg.solve_triangular(L, z, trans="T", lower=True)
        return beta
    
class QRLRTrainer(LRTrainer):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating]):
        # Q, R = QR(X)
        Q, R = jnp.linalg.qr(X)
        # solve R beta = Qty
        beta = jsp.linalg.solve(R, unnorm_covariance(Q, y), lower=False)
        return beta


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
    def fit(cls, X: np.ndarray[(1, 1), np.floating], y: np.ndarray[(1,), np.floating], algo=CholeskyLRTrainer()):
        beta = algo.fit(X, y)
        return LinearRegression(beta = beta)
