#inspired by https://scipy-cookbook.readthedocs.io/items/FittingData.html

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from functools import reduce
from typing import NewType, List, Callable, Union, Iterator, Optional

import unittest

SEED = 12453987
NUMB = 9
MEAN = 0
STD = 0.05

target_funct = lambda x: 0.5+0.4*np.sin(2*np.pi*x)

FitFunct = Callable[[np.ndarray,np.ndarray], np.ndarray] #type alias
deg_0: FitFunct = lambda p, x: x*0+p[0]
deg_1: FitFunct = lambda p, x: p[0]+p[1]*x
deg_3: FitFunct = lambda p, x: p[0]+p[1]*x+p[2]*x**2+p[3]*x**3
deg_9: FitFunct = lambda p, x: (p[0]
                               +p[1]*x**1+p[2]*x**2+p[3]*x**3
                               +p[4]*x**4+p[5]*x**5+p[6]*x**6
                               +p[7]*x**7+p[8]*x**8+p[9]*x**9)

# - f is a python function that takes an ndarray of params
#   and a ndarray of x points returning the y values
# - n_params indicates the number of parameters the function takes
#   in this case the order + 1
# - λ is the regularization punishment for large degrees
# optionally init_params can be used to pass the intial guess to
# start optimising from
def fit_regularized(f: FitFunct, n_params: int, λ: float, 
    x_data: np.ndarray, y_data: np.ndarray, 
    init_params: Optional[np.ndarray] = None) -> np.ndarray:

    if init_params is None:
        #p0 = [0 for i in range(0,n_params)]
        p0 = np.zeros(n_params)
    else:
        p0 = init_params

    err_func = lambda p,x,y: f(p,x)-y - λ/2*square_then_sum(p)
    res = optimize.least_squares(err_func, p0, args=(x_data, y_data))
    return res.x

def square_then_sum(list: Iterator[float]):
    list = map(lambda x: x**2, list)
    return reduce(lambda x, y: x+y, list)

#tests to validate
class Tests(unittest.TestCase):
    def test_upper(self):
        self.assertEqual(square_then_sum([0.5,1,1.5,2]), 7.5)

if __name__ == '__main__':
    #run test
    #unittest.main()
    
    x_data = np.linspace(0, 1, num=NUMB)
    y_data = target_funct(x_data)

    np.random.seed(SEED)
    noise_train = np.random.normal(MEAN, STD, NUMB)
    noise_test = np.random.normal(MEAN, STD, NUMB)

    y_train = y_data+noise_train
    y_test = y_data+noise_test

    fit_funct = deg_9
    fit_params = fit_regularized(fit_funct, 10, 0, x_data, y_train)
    print(fit_params)
    plt.scatter(x_data, y_train, label="Training data", color="tab:orange")

    x_grid = np.linspace(0,1,500)
    plt.plot(x_grid, fit_funct(fit_params, x_grid), label="Fit", color="tab:blue")
    plt.plot(x_grid, target_funct(x_grid), label="Actual/Target", color="tab:olive")

    plt.show()
    plt.close()