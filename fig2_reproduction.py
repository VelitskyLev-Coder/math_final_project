import random
import numpy as np
import matplotlib.pyplot as plt
from population_maps import BevertonHoltModel, TwoConnectedSubpopulationsModel, RickerModel, TwoSubpopulations, \
    GrowthModel


def calculate_population(model: TwoConnectedSubpopulationsModel, populations: TwoSubpopulations,
                         n_steps) -> TwoSubpopulations:
    for _ in range(n_steps):
        populations = model.calculate_next_population(populations)
    return populations


def create_fig_2_common(growth_model_1: GrowthModel, growth_model_2: GrowthModel, min_x: float, max_x: float,
                        min_y: float, max_y: float, title: str) -> None:
    total_population_sizes = []
    total_population1_sizes = []
    total_population2_sizes = []
    size_dif = []
    m_values = np.linspace(0, 1, 1000)
    n_steps = 200

    population_model = TwoConnectedSubpopulationsModel(growth_model_1, growth_model_2, 0)
    initial_condition = TwoSubpopulations(random.random(), random.random())
    dash_line_y = calculate_population(population_model, initial_condition, n_steps).total

    for m in m_values:
        population_model = TwoConnectedSubpopulationsModel(growth_model_1, growth_model_2, m)
        initial_condition = TwoSubpopulations(random.random(), random.random())
        population_after_steps = calculate_population(population_model, initial_condition, n_steps)
        total_population_sizes.append(population_after_steps.total)
        total_population1_sizes.append(population_after_steps.first_population_size)
        total_population2_sizes.append(population_after_steps.second_population_size)
        size_dif.append(population_after_steps.first_population_size-population_after_steps.second_population_size)

    plt.figure(figsize=(6, 6))
    plt.axhline(y=dash_line_y, color='gray', linestyle='--')
    plt.plot(m_values, total_population_sizes, color='Red', linewidth=3, label='Total Population')
    plt.plot(m_values, total_population1_sizes, color='Green', linewidth=3, label='First Population')
    plt.axhline(y=0.4, color='Green', linestyle='--')
    plt.plot(m_values, total_population2_sizes, color='Blue', linewidth=3, label='Second Population')
    plt.axhline(y=0.1, color='Blue', linestyle='--')
    plt.plot(m_values, size_dif, color='Brown', linewidth=3, label='First Second Difference')
    plt.axvline(x=0.0799, color='Black', linestyle='--', label='Max total')
    plt.axvline(x=0.5, color='Black', linestyle='--', label='Max total')
    plt.xlabel("Dispersal rate, m")
    plt.ylabel("Total Population Size")
    plt.title(title)
    plt.legend()

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)



    plt.savefig(f'{title}.pdf', format='pdf')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(total_population1_sizes, total_population2_sizes, color='Green', linewidth=3, label='First Population')

    f_val = list(np.linspace(0, 1, 1000))
    s_val = [0.5 - f for f in f_val]
    plt.plot(f_val, s_val, color='Blue', linewidth=3, label='First Population')
    plt.plot(total_population1_sizes, total_population2_sizes, color='Green', linewidth=3, label='First Population')
    plt.xlabel("Dispersal rate, m")
    plt.ylabel("Total Population Size")
    plt.title(title)
    plt.legend()

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.savefig(f'complex.pdf', format='pdf')
    plt.show()


def fig_2_beverton_holt():
    beverton_holt_model_1 = BevertonHoltModel(1.4)
    beverton_holt_model_2 = BevertonHoltModel(1.1)
    create_fig_2_common(beverton_holt_model_1, beverton_holt_model_2, 0, 1, 0.4, 0.54,
                        "Beverton Holt")

def fig_2_beverton_holt2():
    beverton_holt_model_1 = BevertonHoltModel(1.4)
    beverton_holt_model_2 = BevertonHoltModel(1.1)
    create_fig_2_common(beverton_holt_model_1, beverton_holt_model_2, 0, 1, -0.2, 0.54,
                        "Beverton Holt2")


def fig_2_ricker():
    ricker_model_1 = RickerModel(1.3636)
    ricker_model_2 = RickerModel(1.1)
    create_fig_2_common(ricker_model_1, ricker_model_2, 0, 1, 0.3, 0.45, "Ricker")


def beverton_holt_growth(r1, r2, m, steps):
    beverton_holt_model_1 = BevertonHoltModel(r1)
    beverton_holt_model_2 = BevertonHoltModel(r2)
    population_model = TwoConnectedSubpopulationsModel(beverton_holt_model_1, beverton_holt_model_2, m)
    cur_population = TwoSubpopulations(r1-1, r2-1)
    first_population_points = []
    second_population_points = []
    total_population_points = []

    first_population_points.append(cur_population.first_population_size)
    second_population_points.append(cur_population.second_population_size)
    total_population_points.append(cur_population.total)

    for _ in range(steps):
        cur_population = population_model.calculate_next_population(cur_population)
        first_population_points.append(cur_population.first_population_size)
        second_population_points.append(cur_population.second_population_size)
        total_population_points.append(cur_population.total)

    plt.figure(figsize=(6, 6))
    plt.axhline(y=r1, color='gray', linestyle='--')
    plt.plot(range(steps+1), total_population_points, color='Red', linewidth=3)
    plt.plot(range(steps+1), first_population_points, color='Green', linewidth=3)
    plt.axhline(y=r1-1, color='Green', linestyle='--')
    plt.plot(range(steps+1), second_population_points, color='Blue', linewidth=3)
    plt.axhline(y=r2-1, color='Blue', linestyle='--')
    plt.xlabel("Step")
    plt.ylabel("Population Size")
    plt.title(f'beverton_holt_{r1}_{r2}_{m}')

    plt.xlim(0, steps+1)
    plt.ylim(0, (r1+r2-2)*1.4)

    plt.savefig(f'beverton_holt_{r1}_{r2}_{m}.pdf', format='pdf')
    plt.show()



def main():
    #fig_2_beverton_holt()
    fig_2_beverton_holt2()
    #beverton_holt_growth(1.4, 1.1, 0.2, 25)
    #fig_2_ricker()


if __name__ == '__main__':
    main()
