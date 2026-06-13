from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from plot_generators.common import (
    PLOTS_DIR,
    save_figure,
    simulate_linked_equilibrium_and_yield,
)


GRID_POINTS = 101
SIMULATION_STEPS = 800
OUTPUT_PATH = PLOTS_DIR / "linked_no_take_yield_examples.pdf"

EXAMPLES = (
    {
        "title": r"No-take in $A$",
        "r_a": 1.8685,
        "r_b": 1.0365,
        "m": 0.7586,
        "edge": "A",
    },
    {
        "title": r"No-take in $B$",
        "r_a": 0.8367,
        "r_b": 2.7031,
        "m": 0.9817,
        "edge": "B",
    },
)


def yield_grid(
    r_a: float,
    r_b: float,
    m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    efforts = np.linspace(0, 1, GRID_POINTS)
    effort_a_grid, effort_b_grid = np.meshgrid(efforts, efforts)
    yields = np.zeros_like(effort_a_grid)

    for row in range(GRID_POINTS):
        for column in range(GRID_POINTS):
            _, _, yields[row, column] = simulate_linked_equilibrium_and_yield(
                r_a=r_a,
                r_b=r_b,
                m=m,
                e_a=float(effort_a_grid[row, column]),
                e_b=float(effort_b_grid[row, column]),
                steps=SIMULATION_STEPS,
            )

    return effort_a_grid, effort_b_grid, yields


def grid_maximum(
    effort_a_grid: np.ndarray,
    effort_b_grid: np.ndarray,
    yields: np.ndarray,
) -> tuple[float, float, float]:
    max_index = np.unravel_index(np.nanargmax(yields), yields.shape)
    return (
        float(effort_a_grid[max_index]),
        float(effort_b_grid[max_index]),
        float(yields[max_index]),
    )


def no_take_candidate(
    r_a: float,
    r_b: float,
    m: float,
    edge: str,
) -> tuple[float, float, float]:
    sqrt_r_a = np.sqrt(r_a)
    sqrt_r_b = np.sqrt(r_b)

    def y_from_x(x_value: float) -> float:
        return np.sqrt(
            (1 - (1 - m) * x_value**2)
            / (1 - m + (2 * m - 1) * x_value**2)
        )

    def x_from_y(y_value: float) -> float:
        return np.sqrt(
            (1 - (1 - m) * y_value**2)
            / (1 - m + (2 * m - 1) * y_value**2)
        )

    def solve_root(function, lower: float, upper: float) -> float:
        points = np.linspace(lower, upper, 2000)
        values = np.array([function(float(point)) for point in points])
        valid = np.isfinite(values)
        points = points[valid]
        values = values[valid]

        for left, right, value_left, value_right in zip(
            points[:-1],
            points[1:],
            values[:-1],
            values[1:],
        ):
            if value_left == 0:
                return float(left)
            if value_left * value_right < 0:
                return float(brentq(function, float(left), float(right)))

        raise ValueError("No no-take candidate root found")

    if edge == "A":
        def residual(x_value: float) -> float:
            y_value = y_from_x(x_value)
            population_a = (sqrt_r_a - x_value) / x_value
            growth_a = sqrt_r_a * (sqrt_r_a - x_value)
            growth_b = sqrt_r_b * (sqrt_r_b - y_value)
            pre_fishing_a = (1 - m) * growth_a + m * growth_b
            return population_a - pre_fishing_a

        x = solve_root(residual, 1e-5, sqrt_r_a - 1e-5)
        y = y_from_x(x)
        population_b = (sqrt_r_b - y) / y
        growth_a = sqrt_r_a * (sqrt_r_a - x)
        growth_b = sqrt_r_b * (sqrt_r_b - y)
        pre_fishing_b = m * growth_a + (1 - m) * growth_b
        e_a = 0.0
        e_b = 1 - population_b / pre_fishing_b
    else:
        def residual(y_value: float) -> float:
            x_value = x_from_y(y_value)
            population_b = (sqrt_r_b - y_value) / y_value
            growth_a = sqrt_r_a * (sqrt_r_a - x_value)
            growth_b = sqrt_r_b * (sqrt_r_b - y_value)
            pre_fishing_b = m * growth_a + (1 - m) * growth_b
            return population_b - pre_fishing_b

        y = solve_root(residual, 1e-5, sqrt_r_b - 1e-5)
        x = x_from_y(y)
        population_a = (sqrt_r_a - x) / x
        growth_a = sqrt_r_a * (sqrt_r_a - x)
        growth_b = sqrt_r_b * (sqrt_r_b - y)
        pre_fishing_a = (1 - m) * growth_a + m * growth_b
        e_a = 1 - population_a / pre_fishing_a
        e_b = 0.0

    _, _, yield_value = simulate_linked_equilibrium_and_yield(
        r_a=r_a,
        r_b=r_b,
        m=m,
        e_a=e_a,
        e_b=e_b,
        steps=SIMULATION_STEPS,
    )
    return float(e_a), float(e_b), float(yield_value)


def first_order_candidate_curve(
    r_a: float,
    r_b: float,
    m: float,
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

    population_a = (sqrt_r_a - x_values) / x_values
    population_b = (sqrt_r_b - y_values) / y_values
    growth_a = sqrt_r_a * (sqrt_r_a - x_values)
    growth_b = sqrt_r_b * (sqrt_r_b - y_values)
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


def create_subplot(ax, example: dict[str, float | str]) -> None:
    r_a = float(example["r_a"])
    r_b = float(example["r_b"])
    m = float(example["m"])
    edge = str(example["edge"])
    effort_a_grid, effort_b_grid, yields = yield_grid(r_a, r_b, m)
    candidate_e_a, candidate_e_b, candidate_yield = no_take_candidate(
        r_a,
        r_b,
        m,
        edge,
    )
    grid_e_a, grid_e_b, _ = grid_maximum(effort_a_grid, effort_b_grid, yields)

    heatmap = ax.contourf(
        effort_a_grid,
        effort_b_grid,
        yields,
        levels=30,
        cmap="viridis",
    )
    contour_levels = np.linspace(0, np.nanmax(yields), 8)[1:]
    contours = ax.contour(
        effort_a_grid,
        effort_b_grid,
        yields,
        levels=contour_levels,
        colors="white",
        linewidths=0.7,
        alpha=0.8,
    )
    ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")

    curve_e_a, curve_e_b = first_order_candidate_curve(r_a, r_b, m)
    if len(curve_e_a) > 1:
        ax.plot(
            curve_e_a,
            curve_e_b,
            color="orange",
            linestyle="--",
            linewidth=2.0,
            label="stationarity curve",
            zorder=3,
        )

    ax.scatter(
        [candidate_e_a],
        [candidate_e_b],
        color="red",
        edgecolor="white",
        linewidth=1.2,
        s=70,
        zorder=4,
        label="analytic edge candidate",
    )
    ax.scatter(
        [grid_e_a],
        [grid_e_b],
        marker="x",
        color="black",
        s=70,
        linewidth=2,
        zorder=4,
        label="grid maximum",
    )
    ax.set_title(
        rf"{example['title']}: $r_A={r_a:.3f}$, $r_B={r_b:.3f}$, $m={m:.3f}$"
        "\n"
        rf"analytic $(e_A,e_B)=({candidate_e_a:.3f},{candidate_e_b:.3f})$, "
        rf"$Y={candidate_yield:.3f}$",
        fontsize=9,
    )
    ax.set_xlabel(r"$e_A$")
    ax.set_ylabel(r"$e_B$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper right", fontsize=8)

    return heatmap


def create_plot(output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    heatmap = None
    for ax, example in zip(axes, EXAMPLES):
        heatmap = create_subplot(ax, example)

    if heatmap is not None:
        colorbar = fig.colorbar(heatmap, ax=axes, shrink=0.9)
        colorbar.set_label("equilibrium yield")

    save_figure(fig, output_path, tight_layout=False)
    plt.close(fig)


def main() -> None:
    create_plot(OUTPUT_PATH)


if __name__ == "__main__":
    main()
