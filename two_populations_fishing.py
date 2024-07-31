import numpy as np
from matplotlib import pyplot as plt
from population_maps import TwoConnectedSubpopulationsModelWithFishing, BevertonHoltModel, TwoSubpopulations,\
    RickerModel
import itertools
import seaborn as sns


def create_extinction_map():
    # Model Params
    r_a = 1.2
    r_b = 1.4
    m = 0.3

    # Numeric Simulation Params
    epsilon = 0.0015
    steps = 1000
    n_points = 1000

    model1 = RickerModel(r_a)
    model2 = RickerModel(r_b)
    start_population = TwoSubpopulations(0.5, 0.5)

    extinction_points = []
    survival_points = []

    for _ in range(n_points):
        e1 = float(np.random.rand())
        e2 = float(np.random.rand())
        model = TwoConnectedSubpopulationsModelWithFishing(model1, model2, m, e1, e2)

        population = model.calculate_next_population_in_n_steps(start_population, steps)

        if population.total < epsilon:
            extinction_points.append((e1, e2))
        else:
            survival_points.append((e1, e2))

    plt.figure(figsize=(6, 6))

    # noinspection PyPep8Naming
    def extinction_criteria(e_a, e_b):
        A = -r_a + m * r_a + r_a * r_b - 2 * m * r_a * r_b
        B = -1 + r_a + r_b - m * r_a - m * r_b - r_a * r_b + 2 * m * r_a * r_b
        C = r_a * r_b - 2 * m * r_a * r_b
        D = r_b - m * r_b - r_a * r_b + 2 * m * r_a * r_b
        criteria1 = e_b <= (A * e_a + B) / (C * e_a + D)
        criteria2 = C * e_a+D <= 0
        return criteria1 | criteria2

    # Generate a grid of e_a and e_b values
    e_a_vals = np.linspace(0, 1, 3000)
    e_b_vals = np.linspace(0, 1, 3000)
    e_a_grid, e_b_grid = np.meshgrid(e_a_vals, e_b_vals)

    # Calculate the region defined by the conditions
    region = extinction_criteria(e_a_grid, e_b_grid)

    # Plot the region with transparency
    plt.contourf(e_a_grid, e_b_grid, region.astype(int), levels=[-0.5, 0.5, 1.5], cmap='Blues', alpha=0.5)
    plt.colorbar(ticks=[0, 1], label='Condition satisfied (0 = No, 1 = Yes)')

    # Plot the extinction and survival points
    if extinction_points:
        extinction_points = np.array(extinction_points)
        plt.scatter(extinction_points[:, 0], extinction_points[:, 1], color='red', label='Extinct', s=10)

    if survival_points:
        survival_points = np.array(survival_points)
        plt.scatter(survival_points[:, 0], survival_points[:, 1], color='green', label='Survived', s=10)

    plt.xlabel('$e_A$ (Fishing Effort in Region A)')
    plt.ylabel('$e_B$ (Fishing Effort in Region B)')
    plt.title('Extinction Map')

    # Add text annotations for r_a, r_b, and m
    plt.subplots_adjust(bottom=0.3)  # Adjust bottom margin to make space for text

    plt.text(0.3, 0.1, fr'$r_A={r_a};\;r_B={r_b};\;m={m}$',
             transform=plt.gcf().transFigure,
             fontsize=13,
             verticalalignment='bottom',
             horizontalalignment='center')

    plt.legend(loc='upper left', bbox_to_anchor=(0.8, 1.22))
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.savefig('extinction_map2.pdf', format='pdf')
    plt.show()


def create_total_population_contour():
    beverton_holt_model_1 = BevertonHoltModel(1.2)
    beverton_holt_model_2 = BevertonHoltModel(0.8)
    m = 0.1
    start_population = TwoSubpopulations(0.5, 0.5)
    epsilon = 0.001
    steps = 100

    e_a_vals = np.linspace(0, 1, 200)
    e_b_vals = np.linspace(0, 1, 200)

    # Create a grid to hold the population values
    total_populations = np.zeros((len(e_b_vals), len(e_a_vals)))  # Swap axes

    for i, e_b_val in enumerate(e_b_vals):  # Swap axes in loop
        for j, e_a_val in enumerate(e_a_vals):
            model = TwoConnectedSubpopulationsModelWithFishing(beverton_holt_model_1, beverton_holt_model_2, m, e_a_val,
                                                               e_b_val)
            value = model.calculate_next_population_in_n_steps(start_population, steps).total
            total_populations[i, j] = value

    # Plotting the contour lines
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(e_a_vals, e_b_vals)
    contour = plt.contour(X, Y, total_populations, levels=40, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)

    # Setting the tick labels for better readability
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))

    plt.xlabel('$e_a$ Values')
    plt.ylabel('$e_b$ Values')
    plt.title('Total Population Contour Plot')
    plt.colorbar(contour)
    #plt.savefig('total_contour1.pdf', format='pdf')
    plt.show()


def create_total_fishing_contour():
    beverton_holt_model_1 = BevertonHoltModel(1.4)
    beverton_holt_model_2 = BevertonHoltModel(1.2)
    m = 0.2
    start_population = TwoSubpopulations(0.5, 0.5)
    epsilon = 0.001
    steps = 100

    e_a_vals = np.linspace(0, 1, 300)
    e_b_vals = np.linspace(0, 1, 300)

    # Create a grid to hold the population values
    total_populations = np.zeros((len(e_b_vals), len(e_a_vals)))  # Swap axes

    for i, e_b_val in enumerate(e_b_vals):  # Swap axes in loop
        for j, e_a_val in enumerate(e_a_vals):
            model = TwoConnectedSubpopulationsModelWithFishing(beverton_holt_model_1, beverton_holt_model_2, m, e_a_val,
                                                               e_b_val)
            population = model.calculate_next_population_in_n_steps(start_population, steps)
            value = population.first_population_size/(1-e_a_val)*e_a_val \
                    + population.second_population_size/(1-e_b_val)*e_b_val
            total_populations[i, j] = value

    # Plotting the contour lines
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(e_a_vals, e_b_vals)
    contour = plt.contour(X, Y, total_populations, levels=80, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)

    # Setting the tick labels for better readability
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))

    plt.xlabel('$e_a$ Values')
    plt.ylabel('$e_b$ Values')
    plt.title('Total Fishing Contour Plot')
    plt.colorbar(contour)
   # plt.savefig('total_fishing_contour1.pdf', format='pdf')
    plt.show()


def create_map():
    beverton_holt_model_1 = BevertonHoltModel(1.4)
    beverton_holt_model_2 = BevertonHoltModel(1.2)
    m = 0.2
    start_population = TwoSubpopulations(0.5, 0.5)
    epsilon = 0.001
    steps = 100

    e_a_vals = np.linspace(0.056, 0.057, 100)
    e_b_vals = np.linspace(0.248, 0.249, 100)

    # Create a grid to hold the population values
    total_populations = np.zeros((len(e_b_vals), len(e_a_vals)))  # Swap axes

    N_A_values = []
    N_B_values = []

    for i, e_b_val in enumerate(e_b_vals):  # Swap axes in loop
        for j, e_a_val in enumerate(e_a_vals):
            model = TwoConnectedSubpopulationsModelWithFishing(beverton_holt_model_1, beverton_holt_model_2, m, e_a_val,
                                                               e_b_val)
            eq_population = model.calculate_next_population_in_n_steps(start_population, steps)
            N_A = eq_population.first_population_size
            N_B = eq_population.second_population_size

            N_A_values.append(N_A)
            N_B_values.append(N_B)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(N_A_values, N_B_values, c='blue', marker='o')
    plt.xlabel('N_A')
    plt.ylabel('N_B')
    plt.title('Scatter plot of N_A vs N_B')
    plt.grid(True)
    plt.show()

def main():
    create_extinction_map()
    #create_total_population_contour()
    #create_total_fishing_contour()
    #create_map()

if __name__ == '__main__':
    main()
