import numpy as np
from jax import jit, vmap
from jax import numpy as jnp

class FedLinearRegression:
    
    def __init__(self, beta: np.ndarray[(1,), np.floating]) -> None:
        self.__beta = beta
    
    def predict(self, X: np.ndarray[(1, 1), np.floating]):
        return jnp.dot(X, self.__beta)
    
    @jit
    def vmap_predict(self, X: np.ndarray[(1, 1), np.floating]):
        return vmap(self.predict)(X)
