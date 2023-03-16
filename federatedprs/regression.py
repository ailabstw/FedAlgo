import numpy as np
from .stats import gen_mvdot

class FedLinearRegression:
    
    def __init__(self, beta: np.ndarray[(1,), np.floating]) -> None:
        self.__beta = beta
    
    def predict(self, X: np.ndarray[(1, 1), np.floating]):
        f = gen_mvdot(self.__beta)
        return f(X)
