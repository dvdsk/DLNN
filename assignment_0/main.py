#inspired by https://scipy-cookbook.readthedocs.io/items/FittingData.html

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from typing import NewType, List, Callable

SEED = 12453987
NUMB = 9
MEAN = 0
STD = 0.05

FitFunct = NewType("FitFunct", Callable[[List[float],np.ndarray], np.ndarray])
deg_0: FitFunct = lambda p, x: p[0]
deg_1: FitFunct = lambda p, x: p[0]+p[1]*x
deg_3: FitFunct = lambda p, x: p[0]+p[1]*x+p[2]*x**2+p[3]*x**3
deg_9: FitFunct = lambda p, x: (p[0]
                               +p[1]*x**1+p[2]*x**2+p[3]*x**3
                               +p[4]*x**4+p[5]*x**5+p[6]*x**6
                               +p[7]*x**7+p[8]*x**8+p[9]*x**9)

# - f is a python function that takes an ndarray of params
#   and a ndarray of x points returning the y values
# - n_params indicates the number of parameters the function takes
# - λ is the regularization punishment for large degrees
# optionally init_params can be used to pass the intial guess to
# start optimising from
def fit_regularized(f: FitFunct, n_params: int, λ: float, 
    x_data: np.ndarray, y_data: np.ndarray, init_params=None):

    if init_params is None:
        p0 = [0 for i in range(0,n_params)]
    else:
        p0 = init_params

    err_func = lambda p,x,y: f(p,x)-y - λ/2*sum(p)**2
    p1, success = optimize.leastsq(err_func, p0[:], args=(x_data, y_data))

    return p1



x_data = np.linspace(2.0, 3.0, num=NUMB)
y_data = 0.5+0.4*np.sin(2*np.pi*x_data)

np.random.seed(SEED)
noise_train = np.random.normal(MEAN, STD, NUMB)
noise_test = np.random.normal(MEAN, STD, NUMB)

y_train = y_data+noise_train
y_test = y_data+noise_test

res = fit_regularized(deg_1, 2, 1, x_data, y_train)
print(res)