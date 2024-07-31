import numpy as np
from scipy.optimize import minimize
from math import sqrt

# Define the functions
def f_a(n_a, r_a):
    return r_a * n_a / (1 + n_a)


def f_b(n_b, r_b):
    return r_b * n_b / (1 + n_b)


def calc_n_a_next(n_a, n_b, r_a, r_b, m, e_a):
    return ((1 - m) * f_a(n_a, r_a) + m * f_b(n_b, r_b)) * (1 - e_a)


def calc_n_b_next(n_a, n_b, r_a, r_b, m, e_b):
    return (m * f_a(n_a, r_a) + (1 - m) * f_b(n_b, r_b)) * (1 - e_b)


def g(n_a, n_b, r_a, r_b):
    return f_a(n_a, r_a) + f_b(n_b, r_b) - (n_a + n_b)


# Define the optimization function
def optimization_func(e_vars, r_a, r_b, m):
    e_a, e_b = e_vars
    # Initial guess for n_a and n_b
    n_a_guess = 1.0
    n_b_guess = 1.0
    # Solve for n_a and n_b using fixed-point iteration
    for _ in range(10000):  # Iterate to find a fixed point
        n_a_next = calc_n_a_next(n_a_guess, n_b_guess, r_a, r_b, m, e_a)
        n_b_next = calc_n_b_next(n_a_guess, n_b_guess, r_a, r_b, m, e_b)
        if np.isclose(n_a_guess, n_a_next, rtol=1e-7) and np.isclose(n_b_guess, n_b_next, rtol=1e-7):
            n_a_guess, n_b_guess = n_a_next, n_b_next
            break
        n_a_guess, n_b_guess = n_a_next, n_b_next
    else:
        print("Fixed-point iteration did not converge")

    # Calculate g
    g_val = g(n_a_guess, n_b_guess, r_a, r_b)
    print(n_a_guess, n_b_guess)
    return -g_val  # Minimize the negative of g to maximize g


# Define initial guess and bounds for e_a and e_b
initial_guess = [0, 0]
bounds = [(0, 1), (0, 1)]

# Define values for r_a, r_b, and m
r_a_val = 1.3
r_b_val = 1.26481
m_val = 1

# Perform optimization
result = minimize(optimization_func, initial_guess, args=(r_a_val, r_b_val, m_val), bounds=bounds, tol=1e-6)

# Get the optimal values of e_a and e_b
optimal_e_a, optimal_e_b = result.x
optimal_g = -result.fun

print(f"Optimal e_a: {optimal_e_a}")
print(f"Optimal e_b: {optimal_e_b}")
print(f"Maximum g: {optimal_g}")


def calc(m, r_a, r_b):
    num = m*r_a-m*r_b+m*sqrt(r_b)-m*sqrt(r_a)-r_a-1+2*sqrt(r_a)
    den = m*r_a-m*r_b+m*sqrt(r_b)-m*sqrt(r_a)-r_a+sqrt(r_a)
    print(f'{num=}')
    print(f'{den=}')
    return num/den
