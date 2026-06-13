from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_generators.common import (
    LINKED_K_A,
    LINKED_K_B,
    PLOTS_DIR,
    save_figure,
    simulate_linked_equilibrium_and_yield,
)


R_A = 1.8
R_B = 1.2
M = 0.2
GRID_POINTS = 101
OUTPUT_PATH = PLOTS_DIR / "linked_fishing_yield_ra1_8_rb1_2_m0_2.pdf"
EXTENDED_OUTPUT_PATH = PLOTS_DIR / "linked_fishing_yield_extended_ra1_8_rb1_2_m0_2.pdf"


def interior_candidate(
    r_a: float,
    r_b: float,
    m: float,
    k_a: float = LINKED_K_A,
    k_b: float = LINKED_K_B,
) -> tuple[float, float, float, float]:
    sqrt_r_a = np.sqrt(r_a)
    sqrt_r_b = np.sqrt(r_b)

    population_a = k_a * (sqrt_r_a - 1)
    population_b = k_b * (sqrt_r_b - 1)

    pre_fishing_a = (
        (1 - m) * k_a * sqrt_r_a * (sqrt_r_a - 1)
        + m * k_b * sqrt_r_b * (sqrt_r_b - 1)
    )
    pre_fishing_b = (
        m * k_a * sqrt_r_a * (sqrt_r_a - 1)
        + (1 - m) * k_b * sqrt_r_b * (sqrt_r_b - 1)
    )

    effort_a = 1 - population_a / pre_fishing_a
    effort_b = 1 - population_b / pre_fishing_b
    return population_a, population_b, effort_a, effort_b


def first_order_candidate_curve(
    r_a: float,
    r_b: float,
    m: float,
    k_a: float = LINKED_K_A,
    k_b: float = LINKED_K_B,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_r_a = np.sqrt(r_a)
    sqrt_r_b = np.sqrt(r_b)
    x_values = np.linspace(1e-4, sqrt_r_a - 1e-4, 2000)

    numerator = 1 - (1 - m) * x_values**2
    denominator = 1 - m + (2 * m - 1) * x_values**2
    valid = (numerator >= 0) & (denominator > 0)
    x_values = x_values[valid]
    y_values = np.sqrt(numerator[valid] / denominator[valid])

    valid = (y_values > 0) & (y_values <= sqrt_r_b)
    x_values = x_values[valid]
    y_values = y_values[valid]

    population_a = k_a * (sqrt_r_a - x_values) / x_values
    population_b = k_b * (sqrt_r_b - y_values) / y_values
    growth_a = k_a * sqrt_r_a * (sqrt_r_a - x_values)
    growth_b = k_b * sqrt_r_b * (sqrt_r_b - y_values)
    pre_fishing_a = (1 - m) * growth_a + m * growth_b
    pre_fishing_b = m * growth_a + (1 - m) * growth_b

    effort_a = 1 - population_a / pre_fishing_a
    effort_b = 1 - population_b / pre_fishing_b
    valid = (
        np.isfinite(effort_a)
        & np.isfinite(effort_b)
        & (effort_a >= 0)
        & (effort_a <= 1)
        & (effort_b >= 0)
        & (effort_b <= 1)
    )
    return effort_a[valid], effort_b[valid]


def simulated_yield_grid(
    r_a: float,
    r_b: float,
    m: float,
    effort_min: float,
    effort_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    efforts = np.linspace(effort_min, effort_max, GRID_POINTS)
    effort_a_grid, effort_b_grid = np.meshgrid(efforts, efforts)
    yield_grid = np.zeros_like(effort_a_grid)

    for row in range(GRID_POINTS):
        for column in range(GRID_POINTS):
            try:
                population_a, population_b, yield_value = simulate_linked_equilibrium_and_yield(
                    r_a=r_a,
                    r_b=r_b,
                    m=m,
                    e_a=float(effort_a_grid[row, column]),
                    e_b=float(effort_b_grid[row, column]),
                )
            except ZeroDivisionError:
                yield_grid[row, column] = np.nan
                continue

            if (
                np.isfinite(population_a)
                and np.isfinite(population_b)
                and np.isfinite(yield_value)
                and abs(population_a) < 1e6
                and abs(population_b) < 1e6
                and abs(yield_value) < 1e6
            ):
                yield_grid[row, column] = yield_value
            else:
                yield_grid[row, column] = np.nan

    return effort_a_grid, effort_b_grid, yield_grid


def create_plot(
    r_a: float,
    r_b: float,
    m: float,
    output_path: Path,
    effort_min: float,
    effort_max: float,
    title: str,
    show_grid_maximum: bool,
    show_first_order_curve: bool,
) -> None:
    effort_a_grid, effort_b_grid, yield_grid = simulated_yield_grid(
        r_a,
        r_b,
        m,
        effort_min,
        effort_max,
    )
    _, _, candidate_e_a, candidate_e_b = interior_candidate(r_a, r_b, m)

    finite_yields = yield_grid[np.isfinite(yield_grid)]
    if effort_min == 0 and effort_max == 1:
        contour_min = np.nanmin(finite_yields)
        contour_max = np.nanmax(finite_yields)
    else:
        contour_min = np.nanpercentile(finite_yields, 5)
        contour_max = np.nanpercentile(finite_yields, 95)

    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap = ax.contourf(
        effort_a_grid,
        effort_b_grid,
        yield_grid,
        levels=np.linspace(contour_min, contour_max, 31),
        cmap="viridis",
        extend="both",
    )
    contour_levels = np.linspace(contour_min, contour_max, 9)[1:-1]
    contours = ax.contour(
        effort_a_grid,
        effort_b_grid,
        yield_grid,
        levels=contour_levels,
        colors="white",
        linewidths=0.7,
        alpha=0.8,
    )
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")

    if show_first_order_curve:
        curve_e_a, curve_e_b = first_order_candidate_curve(r_a, r_b, m)
        if len(curve_e_a) > 1:
            ax.plot(
                curve_e_a,
                curve_e_b,
                color="orange",
                linewidth=2.0,
                linestyle="--",
                label="no-take stationarity curve",
                zorder=2,
            )

    if 0 < candidate_e_a < 1 and 0 < candidate_e_b < 1:
        ax.scatter(
            [candidate_e_a],
            [candidate_e_b],
            color="red",
            edgecolor="white",
            linewidth=1.2,
            s=70,
            label="analytic interior candidate",
            zorder=3,
        )

    if show_grid_maximum:
        max_index = np.unravel_index(np.nanargmax(yield_grid), yield_grid.shape)
        ax.scatter(
            [effort_a_grid[max_index]],
            [effort_b_grid[max_index]],
            marker="x",
            color="black",
            s=70,
            linewidth=2,
            label="grid maximum",
            zorder=3,
        )

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("equilibrium yield")
    ax.set_xlabel(r"$e_A$")
    ax.set_ylabel(r"$e_B$")
    ax.set_title(title)
    ax.set_xlim(effort_min, effort_max)
    ax.set_ylim(effort_min, effort_max)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.15)

    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    create_plot(
        r_a=R_A,
        r_b=R_B,
        m=M,
        output_path=OUTPUT_PATH,
        effort_min=0,
        effort_max=1,
        title=rf"Linked fishing yield, $r_A={R_A}$, $r_B={R_B}$, $m={M}$",
        show_grid_maximum=True,
        show_first_order_curve=False,
    )
    create_plot(
        r_a=R_A,
        r_b=R_B,
        m=M,
        output_path=EXTENDED_OUTPUT_PATH,
        effort_min=-10,
        effort_max=10,
        title="Linked fishing yield on extended effort range",
        show_grid_maximum=False,
        show_first_order_curve=True,
    )


if __name__ == "__main__":
    main()
