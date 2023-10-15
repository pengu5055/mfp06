import numpy as np
import matplotlib.pyplot as plt
from diffeq import *  # For some reason te metode pac ne delajo
import scipy.integrate
import matplotlib.colors
from matplotlib.collections import LineCollection
import time

# TODO: Numerical accuracy of different methods
# TODO: Step size and errors of different methods


# Define equation parameters and equation; x' = f(x, t)
def f(x, t):
    return -k*(x - T_out)


def f2(x, t):
    return t*np.sin(t)


def exact(x, k, T_out, T0):
    return (T0 - T_out)*np.exp(-k*x) + T_out


def exact2(x, k , T_out, T0):
    return (T0 - T_out)*np.exp(-k*x) + k*np.sin(t) -k*np.cos(t) + T_out
    # return (T0 - T_out) * np.exp(-k * x) + 0.099099 * np.sin(t) - 0.99099 * np.cos(t) + T_out


def runge_kuta4(f, x0, t):
    h = t[1] - t[0]
    x = x0
    output = []
    for time in t:
        k1 = f(x, time)
        k2 = f(x + h/2 * k1, time + h/2)
        k3 = f(x + h/2 * k2, time + h/2)
        k4 = f(x + h*k3, time + h)
        x += h/6 * (k1 + 2*k2 + 2*k3 + k4)
        output.append(x)

    return np.array(output)


def solve_euler(f, x0, t):
    h = t[1] - t[0]
    x = x0
    output = []
    for time in t:
        x += h*f(x, time)
        output.append(x)

    return np.array(output)


def midpoint(f, x0, t):
    h = t[1] - t[0]
    x = x0
    output = []
    for time in t:
        k1 = h * f(x, time)/2
        x += h * f(x + k1, time + h/2)
        output.append(x)

    return np.array(output)


def solve_heun(f, x0, t):
    h = t[1] - t[0]  # Should be constant anyways
    x = x0
    output = []
    for time in t:
        k1 = h * f(x, time)
        k2 = h * f(x + k1, time + h)
        x += 0.5 * (k1 + k2)
        output.append(x)

    return np.array(output)


def step_avg_error(a, b, method, f, x0, exact, time_start, time_stop, *args):
    """
    Returns average errors of method at different step values.

    INPUTS:
    a: n range start
    b: n range stop
    method: callable function for method to solve ODE
    f: callable function equaling x' = f(x,t) for method
    x0: starting parameter for x(t[0])
    exact: callable function for exact values to calculate error
    time_start: time range start
    time_stop: time range stop

    OUTPUT:
    k: array of step sizes
    output: array of corresponding average errors
    """
    output = []
    n = np.arange(a, b)
    k = []
    for item in n:  # Item is n division for times
        print(item)
        t = np.linspace(time_start, time_stop, item)
        f_method = method(f, x0, t)
        f_exact = exact(t, *args)
        output.append(np.median(np.abs(f_method - f_exact)))
        k.append((time_stop-time_start)/item)

    return np.array(k), np.array(output)


def step_error(a, b, method, f, x0, exact, time_start, time_stop, *args):
    """
    Returns errors of method at different step values.

    INPUTS:
    a: n range start
    b: n range stop
    method: callable function for method to solve ODE
    f: callable function equaling x' = f(x,t) for method
    x0: starting parameter for x(t[0])
    exact: callable function for exact values to calculate error
    time_start: time range start
    time_stop: time range stop

    OUTPUT:
    times: array of corresponding time arrays for output
    output: array of error arrays
    k: array of step sizes
    run_time: array of run time for method call for each step size
    """
    output = []
    n = np.arange(a, b)
    k = []
    times = []
    run_time = []

    for item in n:  # Item is n division for times
        print(item)
        t = np.linspace(time_start, time_stop, item)
        times.append(t)
        pre = time.clock()
        f_method = method(f, x0, t)
        post = time.clock()
        run_time.append(post-pre)
        f_exact = exact(t, *args)
        output.append(np.abs(f_method - f_exact))
        k.append((time_stop-time_start)/item)

    return np.array(times), np.array(output), np.array(k), np.array(run_time)


def parameter_solve_f1(a, b, x0, para_start, para_stop, para_div):  # Can't create line collection??
    """
    Solve ODE2 at different parameter k values with RKF method
    """
    times = []
    output = []
    parameters = np.linspace(para_start, para_stop, para_div)

    for param in parameters:
        def f(x, t):
            return -param * (x - T_out)
        t_rkf, f_rkf = rkf(f, a, b, x0, 10**-6, 1, 0.01)
        times.append(np.ndarray.tolist(t_rkf))
        output.append(np.ndarray.tolist(f_rkf))

    return np.array(times), np.array(output), np.array(parameters)


# def parameter_solve_f1(a, b, n, x0, para_start, para_stop, para_div):
#     """
#     Solve ODE2 at different parameter k values with RKF method
#     """
#     t = np.linspace(a, b, n)
#
#     output = []
#     parameters = np.linspace(para_start, para_stop, para_div)
#
#     for param in parameters:
#         def f(x, t):
#             return -param * (x - T_out)
#         funct = solve_heun(f, x0, t)
#         output.append(funct)
#
#     return t, np.array(output), np.array(parameters)

# Equation parameters
T_out = -5
k = 0.1
T0 = 21
# Interval parameters
a, b = (0.0, 20.0)
n = 1000
t = np.linspace(a, b, n)

# f_euler2 = euler(f, T0, t)
# f_exact = exact(t, k, T_out, T0)
# plt.plot(t, f_euler2, c="#BC23BA", label="Euler")
# plt.plot(t, f_exact, c="#63E649", label="Exact")
# plt.title("Nedelujoča Eulerjeva metoda za h = {}".format((b - a)/n))
# plt.xlabel("Čas [s]")
# plt.ylabel("Temperatura [°C]")
# plt.legend()
# plt.show()
# x0 = 1
# f_euler = solve_euler(f2, x0, t)
# f_midpoint = midpoint(f2, x0, t)
# f_heun = solve_heun(f2, x0, t)
# f_pc4 = pc4(f2, x0, t)  # TODO: Predictor Corrector 4 not working
# f_rku4 = runge_kuta4(f2, x0, t)
# f_scipy_pre = scipy.integrate.odeint(f2, x0, t)  # Returns an array of arrays
# f_scipy = np.array([float(value) for value in np.nditer(f_scipy_pre)])  # Processing into regular array
# t_rkf, f_rkf = rkf(f2, a, b, x0, 10**-6, 1, 0.01)
# f_exact = np.sin(t) - t*np.cos(t) + x0
# f_exact_rkf = np.sin(t_rkf) - t_rkf*np.cos(t_rkf) + x0

# Methods and method errors graph on test function
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.plot(t, f_euler, label="Euler", c="#71B9F8")
# ax1.plot(t, f_heun, label="Heun", c="#6387E9")
# ax1.plot(t, f_midpoint, label="Midpoint", c="#CF26CC")
# ax1.plot(t, f_rku4, label="RK 4", c="#9646DC")
# ax1.plot(t, f_exact,label="Exact", c="#63E649")
# ax1.plot(t_rkf, f_rkf)
# ax1.set_title("Prikaz različnih rešitev pri h = {}".format((b-a)/n))
# ax1.set_xlabel("Čas [s]")
# ax1.set_ylabel("Temperatura [°C]")
# ax1.legend()

# ax2.plot(t, np.abs(f_euler - f_exact), label="Euler", c="#71B9F8")
# ax2.plot(t, np.abs(f_heun - f_exact), label="Heun", c="#6387E9")
# ax2.plot(t, np.abs(f_midpoint - f_exact), label="Midpoint", c="#CF26CC")
# ax2.plot(t, np.abs(f_rku4 - f_exact), label="RK 4", c="#9646DC")

# ax2.set_title("Absolutne napake")
# ax2.set_xlabel("Čas [s]")
# ax2.set_ylabel("Absolutna napaka")
# ax2.set_yscale("log")
# ax2.legend(loc="best")
# plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.plot(t, f_scipy, label="Scipy", c="#BC23BA")
# ax1.plot(t_rkf, f_rkf, label="RKF", c="#DA3E52")
# ax1.plot(t, f_exact, label="Exact", c="#63E649")
# ax1.set_title("Prikaz rešitev adaptivnih metod")
# ax1.set_xlabel("Čas [s]")
# ax1.set_ylabel("Temperatura [°C]")
# ax1.legend()
#
# ax2.plot(t, np.abs(f_scipy - f_exact), label="Scipy", c="#BC23BA")
# ax2.plot(t_rkf, np.abs(f_rkf - f_exact_rkf), label="RKF", c="#DA3E52")
# ax2.set_title("Absolutne napake")
# ax2.set_xlabel("Čas [s]")
# ax2.set_ylabel("Absolutna napaka")
# ax2.set_yscale("log")
# ax2.legend(loc="best")
# plt.show()
#
# f_euler = solve_euler(f, T0, t)
# f_midpoint = midpoint(f, T0, t)
# f_heun = solve_heun(f, T0, t)
# f_pc4 = pc4(f, T0, t)  # TODO: Predictor Corrector 4 not working
# f_rku4 = runge_kuta4(f, T0, t)
# f_scipy_pre = scipy.integrate.odeint(f, T0, t)  # Returns an array of arrays
# f_scipy = np.array([float(value) for value in np.nditer(f_scipy_pre)])  # Processing into regular array
# t_rkf, f_rkf = rkf(f, a, b, T0, 10**-6, 1, 0.01)
# f_exact = exact(t, k, T_out, T0)
# f_exact_rkf = exact(t_rkf, k, T_out, T0)
#
# # Methods and method errors graph
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.plot(t, f_euler, label="Euler", c="#71B9F8")
# ax1.plot(t, f_heun, label="Heun", c="#6387E9")
# ax1.plot(t, f_midpoint, label="Midpoint", c="#CF26CC")
# ax1.plot(t, f_rku4, label="RK 4", c="#9646DC")
# ax1.plot(t, f_exact,label="Exact", c="#63E649")
# ax1.plot(t_rkf, f_rkf)
# ax1.set_title("Prikaz različnih rešitev pri h = {}".format((b-a)/n))
# ax1.set_xlabel("Čas [s]")
# ax1.set_ylabel("Temperatura [°C]")
# ax1.legend()
#
# # Errors graph
# ax2.plot(t, np.abs(f_euler - f_exact), label="Euler", c="#71B9F8")
# ax2.plot(t, np.abs(f_heun - f_exact), label="Heun", c="#6387E9")
# ax2.plot(t, np.abs(f_midpoint - f_exact), label="Midpoint", c="#CF26CC")
# ax2.plot(t, np.abs(f_rku4 - f_exact), label="RK 4", c="#9646DC")
#
# ax2.set_title("Absolutne napake")
# ax2.set_xlabel("Čas [s]")
# ax2.set_ylabel("Absolutna napaka")
# ax2.set_yscale("log")
# ax2.legend(loc="best")
# plt.show()

# Advanced addaptive methods and their errors

# ax1.plot(t, f_scipy, label="Scipy", c="#BC23BA")
# ax1.plot(t_rkf, f_rkf, label="RKF", c="#DA3E52")
# ax1.set_title("Prikaz rešitev adaptivnih metod")
# ax1.set_xlabel("Čas [s]")
# ax1.set_ylabel("Temperatura [°C]")
# ax1.legend()
#
# ax2.plot(t, np.abs(f_scipy - f_exact), label="Scipy", c="#BC23BA")
# ax2.plot(t_rkf, np.abs(f_rkf - f_exact_rkf), label="RKF", c="#DA3E52")
# ax2.set_title("Absolutne napake")
# ax2.set_xlabel("Čas [s]")
# ax2.set_ylabel("Absolutna napaka")
# ax2.set_yscale("log")
# ax2.legend(loc="best")
# plt.show()

# Plot average error of various methods with decreasing step size
# n_euler, averages_euler = step_avg_error(50, 1000, solve_euler, f, T0, exact, 0, 20, k, T0, T_out)
# n_heun, averages_heun = step_avg_error(50, 1000, solve_heun, f, T0, exact, 0, 20, k, T0, T_out)
# n_midpoint, averages_midpoint = step_avg_error(50, 1000, midpoint, f, T0, exact, 0, 20, k, T0, T_out)
# n_rk4, averages_rk4 = step_avg_error(50, 1000, runge_kuta4, f, T0, exact, 0, 20, k, T0, T_out)
# # n_scipy, averages_scipy = step_error(50, 5000, scipy.integrate.odeint, f, T0, exact, 0, 20, k, T0, T_out)
# plt.plot(n_euler, averages_euler, label="Euler")
# plt.plot(n_midpoint, averages_midpoint, label="Midpoint")
# plt.plot(n_rk4, averages_rk4, label="RK 4")
# plt.legend()
# plt.show()

# Plot errors for method at different step sizes (and try and make it pretty :)) )
# TODO: Change this in to plot per method for errors and 1 singular plot for all run times
fig, ax = plt.subplots()
t_test, err_test, k_test, run_times = step_error(5, 10000, solve_euler, f, T0, exact, 0, 20, k, T_out, T0)
t_test1, err_test1, k_test1, run_times1 = step_error(5, 10000, solve_heun, f, T0, exact, 0, 20, k, T_out, T0)
t_test2, err_test2, k_test2, run_times2 = step_error(5, 10000, midpoint, f, T0, exact, 0, 20, k, T_out, T0)
t_test3, err_test3, k_test3, run_times3 = step_error(5, 10000, runge_kuta4, f, T0, exact, 0, 20, k, T_out, T0)
# num_colors = err_test.shape[0]
# segments = [np.column_stack((x, y)) for x, y in zip(t_test, err_test)]
# line_segments = LineCollection(segments, cmap="plasma", norm=matplotlib.colors.LogNorm(vmin=k_test[num_colors - 1], vmax=k_test[0]))
# line_segments.set_array(k_test)
# ax.add_collection(line_segments)
# plt.title("Absolutna napaka RK4 metode")
# plt.colorbar(mappable=line_segments, label="Velikost koraka h")
# plt.xlabel("Čas [s]")
# plt.xlim(0, 20)
# plt.ylabel("Absolutna napaka")
# plt.yscale("log")
# plt.show()

plt.plot(k_test, run_times, c="#71B9F8", label="Euler")
plt.plot(k_test1, run_times1, c="#6387E9", label="Heun")
plt.plot(k_test2, run_times2, c="#CF26CC", label="Midpoint")
plt.plot(k_test3, run_times3, c="#9646DC", label="RK4")

plt.title("Čas računanja različnih metod")
plt.xlabel("Velikost koraka h")
plt.xscale("log")
plt.ylabel("Čas računanja [s]")
# fig.subplots_adjust(left=0.11, right=0.98, wspace=0.28)
plt.legend()
plt.show()

# plt.scatter(k_test, run_times, c="#BC23BA", s=3)
# plt.title("Čas računanja Mitpoint metode")
# plt.xlabel("Velikost koraka h")
# plt.ylabel("Čas računanja [s]")
# # fig.subplots_adjust(left=0.08, right=0.96, wspace=0.41)
# plt.show()

# Plot for solutions at different parameters k
# fig, ax = plt.subplots()
# p_start = 0.01
# p_end = 1
#
# times, curves, k_value = parameter_solve_f1(a=0, b=100, x0=-15, para_start=p_start, para_stop=p_end, para_div=2000)
# # dim = times.shape[0]
# segments = [np.column_stack((x, y)) for x, y in zip(times, curves)]
# col = LineCollection(segments, cmap="plasma", norm=matplotlib.colors.LogNorm(vmin=p_start, vmax=p_end))
# col.set_array(k_value)
# ax.add_collection(col, autolim=True)
# plt.colorbar(mappable=col, label="Vrednost parametra k")
# plt.xlim(0, 100)
# #plt.ylim(-6, 25)
# plt.ylim(-17, -4)
# plt.title("Družina rešitev za drugi začetni pogoj")
# plt.xlabel("Čas [s]")
# plt.ylabel("Temperatura [°C]")
# plt.show()
