from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_generators.common import (
    LINKED_EXTINCTION_EPSILON,
    PLOTS_DIR,
    linked_fishing_spectral_radius,
    save_figure,
    simulate_linked_total_population,
)


GRID_POINTS = 250
RANDOM_POINTS = 1000
RANDOM_SEED = 20260613


@dataclass(frozen=True)
class ExtinctionMapParameters:
    r_a: float
    r_b: float
    m: float

    @property
    def output_path(self) -> Path:
        def encode(value: float) -> str:
            return str(value).replace(".", "_")

        return (
            PLOTS_DIR
            / f"linked_extinction_map_ra{encode(self.r_a)}_rb{encode(self.r_b)}_m{encode(self.m)}.pdf"
        )


PARAMETER_SETS = (
    ExtinctionMapParameters(r_a=1.4, r_b=0.8, m=0.3),
    ExtinctionMapParameters(r_a=1.4, r_b=1.2, m=0.1),
    ExtinctionMapParameters(r_a=1.4, r_b=1.2, m=0.3),
    ExtinctionMapParameters(r_a=1.4, r_b=1.2, m=0.0),
    ExtinctionMapParameters(r_a=1.4, r_b=1.2, m=1.0),
    ExtinctionMapParameters(r_a=1.8, r_b=1.2, m=0.1),
)


def theoretical_extinction_grid(
    parameters: ExtinctionMapParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e_a_values = np.linspace(0, 1, GRID_POINTS)
    e_b_values = np.linspace(0, 1, GRID_POINTS)
    e_a_grid, e_b_grid = np.meshgrid(e_a_values, e_b_values)
    spectral_radius = linked_fishing_spectral_radius(
        e_a=e_a_grid,
        e_b=e_b_grid,
        r_a=parameters.r_a,
        r_b=parameters.r_b,
        m=parameters.m,
    )
    extinction_region = spectral_radius < 1
    return e_a_grid, e_b_grid, spectral_radius, extinction_region


def denominator_zero_effort(parameters: ExtinctionMapParameters) -> float | None:
    c = parameters.r_a * parameters.r_b * (1 - 2 * parameters.m)
    d = parameters.r_b * (
        1 - parameters.m - parameters.r_a * (1 - 2 * parameters.m)
    )

    if np.isclose(c, 0):
        return None

    effort = -d / c
    if 0 <= effort <= 1:
        return effort

    return None


def simulated_points(
    parameters: ExtinctionMapParameters,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    efforts = rng.random((RANDOM_POINTS, 2))
    extinct = np.zeros(RANDOM_POINTS, dtype=bool)

    for index, (e_a, e_b) in enumerate(efforts):
        total_population = simulate_linked_total_population(
            r_a=parameters.r_a,
            r_b=parameters.r_b,
            m=parameters.m,
            e_a=float(e_a),
            e_b=float(e_b),
        )
        extinct[index] = total_population < LINKED_EXTINCTION_EPSILON

    return efforts[:, 0], efforts[:, 1], extinct


def plot_single_map(
    ax,
    parameters: ExtinctionMapParameters,
    rng: np.random.Generator,
) -> None:
    e_a_grid, e_b_grid, spectral_radius, extinction_region = (
        theoretical_extinction_grid(parameters)
    )
    sim_e_a, sim_e_b, sim_extinct = simulated_points(parameters, rng)

    ax.contourf(
        e_a_grid,
        e_b_grid,
        extinction_region.astype(int),
        levels=[-0.5, 0.5, 1.5],
        colors=["white", "#9ecae1"],
        alpha=0.55,
    )
    contour = ax.contour(
        e_a_grid,
        e_b_grid,
        spectral_radius,
        levels=[1],
        colors=["black"],
        linewidths=2,
    )
    ax.clabel(contour, fmt={1: r"$\rho(M_f)=1$"}, inline=True)
    ax.scatter(
        sim_e_a[~sim_extinct],
        sim_e_b[~sim_extinct],
        color="green",
        s=7,
        label="survival",
    )
    ax.scatter(
        sim_e_a[sim_extinct],
        sim_e_b[sim_extinct],
        color="red",
        s=7,
        label="extinction",
    )
    split_effort = denominator_zero_effort(parameters)
    if split_effort is not None:
        ax.axvline(
            split_effort,
            color="black",
            linestyle=":",
            linewidth=2,
            label=r"$Ce_A+D=0$",
        )

    ax.set_title(
        rf"$r_A={parameters.r_a}$, $r_B={parameters.r_b}$, $m={parameters.m}$"
    )
    ax.set_xlabel(r"$e_A$")
    ax.set_ylabel(r"$e_B$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def create_plot(parameters: ExtinctionMapParameters, output_path: Path) -> None:
    rng = np.random.default_rng(
        RANDOM_SEED
        + int(parameters.r_a * 1000)
        + int(parameters.r_b * 100)
        + int(parameters.m * 10)
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_single_map(ax, parameters, rng)
    ax.legend(loc="upper right")

    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    for parameters in PARAMETER_SETS:
        create_plot(parameters, parameters.output_path)


if __name__ == "__main__":
    main()
