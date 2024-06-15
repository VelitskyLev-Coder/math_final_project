import numpy as np
from matplotlib import pyplot as plt
from population_maps import TwoConnectedSubpopulationsModelWithFishing, BevertonHoltModel, TwoSubpopulations

def create_extinction_map():
    beverton_holt_model_1 = BevertonHoltModel(1.4)
    beverton_holt_model_2 = BevertonHoltModel(1.1)
    m = 0.1
    epsilon = 0.001
    steps = 5000
    start_population = TwoSubpopulations(0.5, 0.5)
    extinction_points = []
    survival_points = []

    n_points = 300

    for _ in range(n_points):
        e1 = np.random.rand()
        e2 = np.random.rand()
        model = TwoConnectedSubpopulationsModelWithFishing(beverton_holt_model_1, beverton_holt_model_2, m, e1, e2)

        population = model.calculate_next_population_in_n_steps(start_population, steps)

        if population.total < epsilon:
            extinction_points.append((e1, e2))
        else:
            survival_points.append((e1, e2))

    plt.figure(figsize=(6, 6))

    def lambda_values(e_a, e_b):
        sqrt_term = np.sqrt(15876 * e_a ** 2 - 24332 * e_a * e_b - 7420 * e_a + 9801 * e_b ** 2 + 4730 * e_b + 1345)
        lambda1 = 9 / 8 - (99 * e_b) / 200 - sqrt_term / 200 - (63 * e_a) / 100
        lambda2 = sqrt_term / 200 - (99 * e_b) / 200 - (63 * e_a) / 100 + 9 / 8
        return np.abs(lambda1), np.abs(lambda2)

    # Generate a grid of e_a and e_b values
    e_a_vals = np.linspace(0, 1, 400)
    e_b_vals = np.linspace(0, 1, 400)
    e_a_grid, e_b_grid = np.meshgrid(e_a_vals, e_b_vals)

    # Calculate lambda1 and lambda2 for each pair of (e_a, e_b)
    abs_lambda1, abs_lambda2 = lambda_values(e_a_grid, e_b_grid)

    # Determine the maximum absolute values of lambda1 and lambda2
    max_abs_lambda = np.maximum(abs_lambda1, abs_lambda2)

    # Find the region where the maximum absolute values are less than 1
    region = max_abs_lambda < 1

    # Plot the region with transparency
    plt.contourf(e_a_grid, e_b_grid, region.astype(int), levels=[-0.5, 0.5, 1.5], cmap='Blues', alpha=0.5)
    plt.colorbar(ticks=[0, 1], label='Condition satisfied (0 = No, 1 = Yes)')

    # Plot the extinction and survival points
    if extinction_points:
        extinction_points = np.array(extinction_points)
        plt.scatter(extinction_points[:, 0], extinction_points[:, 1], color='red', label='Extinct')
    if survival_points:
        survival_points = np.array(survival_points)
        plt.scatter(survival_points[:, 0], survival_points[:, 1], color='green', label='Survived')

    plt.xlabel('$e_1$ (Fishing Effort in Region A)')
    plt.ylabel('$e_2$ (Fishing Effort in Region B)')
    plt.title('Extinction Map')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('extinction_map2.pdf', format='pdf')
    plt.show()

def main():
    create_extinction_map()

if __name__ == '__main__':
    main()
