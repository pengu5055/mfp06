import numpy as np
import matplotlib.pyplot as plt
from diffeq import *
import scipy.integrate


def f(x, t):
    return t*np.sin(t)


a, b = (0.0, 20.0)
x0 = 1.0

n = 2000
t = np.linspace(a, b, n)
exact = np.sin(t) - t*np.cos(t) + x0
sci_py = scipy.integrate.odeint(f, x0, t)
x_solve = euler(f, x0, t)
x_solve2 = rku4(f, x0, t)
plt.plot(t, x_solve, label="Euler")
plt.plot(t, x_solve2, label="RK4")
plt.plot(t, exact, label="Exact")
plt.plot(t, sci_py, label="SciPy")
plt.legend()
plt.show()
