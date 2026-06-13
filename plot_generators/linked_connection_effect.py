from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beverton_holt import BevertonHoltModel
from linked_model import LinkedBevertonHoltModel, TwoPopulations
from plot_generators.common import PLOTS_DIR, save_figure


R_A = 1.8
R_B = 1.2
K_A = 1
K_B = 1
M_POINTS = 201
INITIAL_POPULATIONS = TwoPopulations(0.5, 0.5)
OUTPUT_PATH = PLOTS_DIR / "linked_connection_effect_ra1_8_rb1_2.pdf"


def equilibria_over_m() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_values = np.linspace(0, 1, M_POINTS)
    population_a = np.zeros_like(m_values)
    population_b = np.zeros_like(m_values)

    for index, m in enumerate(m_values):
        model = LinkedBevertonHoltModel(
            population_a=BevertonHoltModel(R_A, K_A),
            population_b=BevertonHoltModel(R_B, K_B),
            migration_rate=float(m),
        )
        equilibrium = model.equilibrium(INITIAL_POPULATIONS, max_steps=50000)
        population_a[index] = equilibrium.a
        population_b[index] = equilibrium.b

    return m_values, population_a, population_b


def create_plot(output_path: Path = OUTPUT_PATH) -> None:
    m_values, population_a, population_b = equilibria_over_m()
    total_population = population_a + population_b
    independent_total = (R_A - 1) + (R_B - 1)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(
        m_values,
        total_population,
        color="red",
        linewidth=2.5,
        label=r"$N_A^*+N_B^*$",
    )
    ax.plot(
        m_values,
        population_a,
        color="green",
        linewidth=2,
        label=r"$N_A^*$",
    )
    ax.plot(
        m_values,
        population_b,
        color="blue",
        linewidth=2,
        label=r"$N_B^*$",
    )
    ax.axhline(
        independent_total,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="total at m=0",
    )

    ax.set_title(rf"Connection effect, $r_A={R_A}$, $r_B={R_B}$")
    ax.set_xlabel(r"movement rate $m$")
    ax.set_ylabel("equilibrium population")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    create_plot()


if __name__ == "__main__":
    main()
