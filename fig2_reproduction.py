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
    m_values = np.linspace(0, 1, 1000)
    n_steps = 200

    population_model = TwoConnectedSubpopulationsModel(growth_model_1, growth_model_2, 0)
    initial_condition = TwoSubpopulations(random.random(), random.random())
    dash_line_y = calculate_population(population_model, initial_condition, n_steps).total

    for m in m_values:
        population_model = TwoConnectedSubpopulationsModel(growth_model_1, growth_model_2, m)
        initial_condition = TwoSubpopulations(random.random(), random.random())
        total_population_sizes.append(calculate_population(population_model, initial_condition, n_steps).total)

    plt.figure(figsize=(6, 6))
    plt.plot(m_values, total_population_sizes, color='Red', linewidth=3)
    plt.xlabel("Dispersal rate, m")
    plt.ylabel("Total Population Size")
    plt.title(title)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.axhline(y=dash_line_y, color='gray', linestyle='--')

    plt.savefig(f'{title}.pdf', format='pdf')
    plt.show()


def fig_2_beverton_holt():
    beverton_holt_model_1 = BevertonHoltModel(1.4)
    beverton_holt_model_2 = BevertonHoltModel(1.1)
    create_fig_2_common(beverton_holt_model_1, beverton_holt_model_2, 0, 1, 0.4, 0.54,
                        "Beverton Holt")


def fig_2_ricker():
    ricker_model_1 = RickerModel(1.3636)
    ricker_model_2 = RickerModel(1.1)
    create_fig_2_common(ricker_model_1, ricker_model_2, 0, 1, 0.3, 0.45, "Ricker")


def main():
    fig_2_beverton_holt()
    fig_2_ricker()


if __name__ == '__main__':
    main()
