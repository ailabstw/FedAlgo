from abc import ABCMeta
from typing import Any

import numpy as np
from jax import numpy as jnp
from jax import scipy as jsp
from .linalg import gen_mvmul, mvmul
from .stats import unnorm_autocovariance, unnorm_covariance


class LinearSolver(object, metaclass=ABCMeta):
    def __init__(self) -> None:
        pass
    
class InverseSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        inv_X = jnp.linalg.inv(X)
        # beta = X^-1 y
        return mvmul(inv_X, y)
    
class CholeskySolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        # L = Cholesky(X)
        L = jnp.linalg.cholesky(X)
        # solve Lz = y
        z = jsp.linalg.solve_triangular(L, y, lower=True)
        # solve Lt beta = z
        return jsp.linalg.solve_triangular(L, z, trans="T", lower=True)
    
class QRSolver(LinearSolver):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        # Q, R = QR(X)
        Q, R = jnp.linalg.qr(X)
        # solve R beta = Qty
        return jsp.linalg.solve(R, unnorm_covariance(Q, y), lower=False)


class LinearModel(object, metaclass=ABCMeta):
    def __init__(self) -> None:
        raise NotImplementedError
    
    def predict(self, X: 'np.ndarray[(1, 1), np.floating]'):
        raise NotImplementedError


class LinearRegression(LinearModel):
    """A class for federated linear regression

    Args:
        beta ('np.ndarray[(1,), np.floating]', optional): _description_. Defaults to None.
        XtX ('np.ndarray[(1, 1), np.floating]', optional): _description_. Defaults to None.
        Xty ('np.ndarray[(1,), np.floating]', optional): _description_. Defaults to None.
    """
    
    def __init__(self, beta = None, XtX = None, Xty = None, algo=CholeskySolver()) -> None:
        if beta is None:
            if XtX is None or Xty is None:
                raise ValueError("Must provide XtX and Xty, since beta is not provided.")
            
            if isinstance(algo, QRSolver):
                raise ValueError("QRSolver is not supported in constructor.")
            
            self.__beta = algo(XtX, Xty)
        else:
            self.__beta = beta
        
    @property
    def coef(self):
        return self.__beta
    
    def dof(self, nobs):
        """Degrees of freedom

        Args:
            nobs (int, np.ndarray): Number of observations

        Returns:
            int: _description_
        """
        k = self.__beta.shape[0]
        return nobs - k
    
    def predict(self, X: 'np.ndarray[(1, 1), np.floating]'):
        f = gen_mvmul(self.__beta)
        return f(X)

    @classmethod
    def fit(cls, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]', algo=CholeskySolver()):
        if isinstance(algo, QRSolver):
            beta = algo(X, y)
        else:
            beta = algo(unnorm_autocovariance(X), unnorm_covariance(X, y))
        return LinearRegression(beta = beta)
    
    def residual(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        return y - self.predict(X)
    
    def sse(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        res = self.residual(X, y)
        return jnp.vdot(res.T, res)

    def t_stats(self, sse, XtX, dof):
        XtXinv = jnp.linalg.inv(XtX)
        sigma_squared = sse / dof
        vars = (sigma_squared * XtXinv).diagonal()
        std = jnp.sqrt(vars)
        t_stat = self.coef / std
        return t_stat


class LogisticRegression(LinearModel):
    def __init__(self, beta = None) -> None:
        self.__beta = beta
    
    def predict(self, X: 'np.ndarray[(1, 1), np.floating]'):
        predicted_y = 1 / (1 + jnp.exp(-jnp.dot(X, self.__beta)))
        return jnp.expand_dims(predicted_y, -1)
    
    def fit(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        grad = self.gradient(X, y)
        H = self.hessian(X)
        self.__beta = self.beta(grad, H)
        
    def residual(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        return jnp.expand_dims(y, -1) - self.predict(X)
    
    def gradient(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        return mvmul(X.T, self.residual(X, y))
    
    def hessian(self, X: 'np.ndarray[(1, 1), np.floating]'):
        predicted_y = self.predict(X)
        return jnp.dot(jnp.multiply(X.T, (predicted_y * (1 - predicted_y)).T), X)
    
    def loglikelihood(self, X: 'np.ndarray[(1, 1), np.floating]', y: 'np.ndarray[(1,), np.floating]'):
        epsilon = jnp.finfo(float).eps
        predicted_y = self.predict(X)
        return jnp.sum(y * jnp.log(predicted_y + epsilon) + (1 - y) * jnp.log(1 - predicted_y + epsilon))
    
    def beta(self, gradient, hessian, solver=CholeskySolver()):
        # solver calculates H^-1 grad in faster way
        return jnp.expand_dims(self.__beta, -1) + solver(hessian, gradient)
    