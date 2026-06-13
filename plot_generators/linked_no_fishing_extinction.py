from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_generators.common import (
    LINKED_EXTINCTION_EPSILON,
    PLOTS_DIR,
    linked_spectral_radius,
    save_figure,
    simulate_linked_total_population,
)


GRID_POINTS = 300
RANDOM_POINTS = 1000
RANDOM_SEED = 20260613
M = 0.2
OUTPUT_PATH = PLOTS_DIR / "linked_no_fishing_extinction_m0_2.pdf"


def simulated_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    rates = 3 * rng.random((RANDOM_POINTS, 2))
    extinct = np.zeros(RANDOM_POINTS, dtype=bool)

    for index, (r_a, r_b) in enumerate(rates):
        total_population = simulate_linked_total_population(
            r_a=float(r_a),
            r_b=float(r_b),
            m=M,
            e_a=0,
            e_b=0,
        )
        extinct[index] = total_population < LINKED_EXTINCTION_EPSILON

    return rates[:, 0], rates[:, 1], extinct


def create_plot(output_path: Path = OUTPUT_PATH) -> None:
    r_a_values = np.linspace(0, 3, GRID_POINTS)
    r_b_values = np.linspace(0, 3, GRID_POINTS)
    r_a_grid, r_b_grid = np.meshgrid(r_a_values, r_b_values)

    spectral_radius = linked_spectral_radius(r_a_grid, r_b_grid, M)
    extinction_region = spectral_radius < 1
    sim_r_a, sim_r_b, sim_extinct = simulated_points()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(
        r_a_grid,
        r_b_grid,
        extinction_region.astype(int),
        levels=[-0.5, 0.5, 1.5],
        colors=["white", "#9ecae1"],
        alpha=0.7,
    )
    contour = ax.contour(
        r_a_grid,
        r_b_grid,
        spectral_radius,
        levels=[1],
        colors=["black"],
        linewidths=2,
    )
    ax.clabel(contour, fmt={1: r"$\rho(M)=1$"}, inline=True)
    ax.scatter(
        sim_r_a[~sim_extinct],
        sim_r_b[~sim_extinct],
        color="green",
        s=7,
        label="survival",
    )
    ax.scatter(
        sim_r_a[sim_extinct],
        sim_r_b[sim_extinct],
        color="red",
        s=7,
        label="extinction",
    )

    ax.set_title(rf"Extinction without fishing, $m={M}$")
    ax.set_xlabel(r"$r_A$")
    ax.set_ylabel(r"$r_B$")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    create_plot()


if __name__ == "__main__":
    main()
