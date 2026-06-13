import numpy as np
from scipy.optimize import fsolve


# Define the system of equations
def equations(vars):
    N_a, N_b, e_a, e_b, lambda1, lambda2 = vars
    r_a = 1.5  # Replace with actual value if different
    r_b = 1.4  # Replace with actual value if different
    m = 0.7    # Replace with actual value if different

    eq1 = -N_a * r_a - lambda1 * (r_a * (e_a - 1) * (m - 1) - (N_a + 1)**2) + lambda2 * m * r_a * (e_b - 1) + r_a * (N_a + 1) - (N_a + 1)**2
    eq2 = -N_b * r_b + lambda1 * m * r_b * (e_a - 1) - lambda2 * (r_b * (e_b - 1) * (m - 1) - (N_b + 1)**2) + r_b * (N_b + 1) - (N_b + 1)**2
    eq3 = -lambda1 * (N_a * r_a * (N_b + 1) * (m - 1) - N_b * m * r_b * (N_a + 1))
    eq4 = lambda2 * (N_a * m * r_a * (N_b + 1) - N_b * r_b * (N_a + 1) * (m - 1))
    eq5 = N_a * (N_a + 1) * (N_b + 1) - (e_a - 1) * (N_a * r_a * (N_b + 1) * (m - 1) - N_b * m * r_b * (N_a + 1))
    eq6 = N_b * (N_a + 1) * (N_b + 1) + (e_b - 1) * (N_a * m * r_a * (N_b + 1) - N_b * r_b * (N_a + 1) * (m - 1))

    return [eq1, eq2, eq3, eq4, eq5, eq6]

# Initial guesses for N_a, N_b, r_a, r_b, lambda1, lambda2
initial_guesses = [1, 1, 0.1, 0.1, 1, 1]

# Solve the system of equations
solution = fsolve(equations, initial_guesses)
N_a, N_b, e_a, e_b, lambda1, lambda2 = solution
# Print the solution

print(e_a, e_b, e_a*N_a/(1-e_a)+e_b*N_b/(1-e_b))
print('---')
print(f'{N_a=}')
print(f'{N_b=}')
print(f'{e_a=}')
print(f'{e_b=}')
print(f'{lambda1=}')
print(f'{lambda2=}')