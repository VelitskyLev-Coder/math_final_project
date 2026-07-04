from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from os import cpu_count
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from plot_generators.common import (
    LINKED_K_A,
    LINKED_K_B,
    PLOTS_DIR,
    save_figure,
    simulate_linked_equilibrium_and_yield,
)


CaseKind = Literal[
    "interior",
    "complete_a",
    "complete_b",
    "source_b_sink_a",
    "source_a_sink_b",
    "no_take_a",
    "no_take_b",
]


GRID_POINTS = 91
HIGH_RES_GRID_POINTS = 301
SIMULATION_STEPS = 700
WORKER_COUNT = max(1, (cpu_count() or 2) - 1)


@dataclass(frozen=True)
class Example:
    title: str
    r_a: float
    r_b: float
    m: float
    kind: CaseKind


@dataclass(frozen=True)
class EffortGridResult:
    effort_a_grid: np.ndarray
    effort_b_grid: np.ndarray
    yields: np.ndarray
    max_effort_a: float
    max_effort_b: float
    max_population_a: float
    max_population_b: float
    max_yield: float


def growth(population: np.ndarray | float, r: float, k: float) -> np.ndarray | float:
    return k * r * population / (k + population)


def pre_fishing_growth(
    population_a: np.ndarray | float,
    population_b: np.ndarray | float,
    r_a: float,
    r_b: float,
    m: float,
    k_a: float = LINKED_K_A,
    k_b: float = LINKED_K_B,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    growth_a = growth(population_a, r_a, k_a)
    growth_b = growth(population_b, r_b, k_b)
    return (
        (1 - m) * growth_a + m * growth_b,
        m * growth_a + (1 - m) * growth_b,
    )


def population_yield(
    population_a: np.ndarray | float,
    population_b: np.ndarray | float,
    r_a: float,
    r_b: float,
    k_a: float = LINKED_K_A,
    k_b: float = LINKED_K_B,
) -> np.ndarray | float:
    return (
        growth(population_a, r_a, k_a)
        + growth(population_b, r_b, k_b)
        - population_a
        - population_b
    )


def recover_efforts(
    population_a: float,
    population_b: float,
    r_a: float,
    r_b: float,
    m: float,
    k_a: float = LINKED_K_A,
    k_b: float = LINKED_K_B,
) -> tuple[float, float]:
    pre_a, pre_b = pre_fishing_growth(
        population_a,
        population_b,
        r_a,
        r_b,
        m,
        k_a,
        k_b,
    )
    effort_a = 1 - population_a / pre_a if pre_a > 0 else 0.0
    effort_b = 1 - population_b / pre_b if pre_b > 0 else 0.0
    return float(effort_a), float(effort_b)


def interior_candidate(example: Example) -> tuple[float, float]:
    return (
        LINKED_K_A * (np.sqrt(example.r_a) - 1),
        LINKED_K_B * (np.sqrt(example.r_b) - 1),
    )


def complete_harvest_candidate(example: Example) -> tuple[float, float]:
    if example.kind == "complete_a":
        return 0.0, LINKED_K_B * (np.sqrt(example.r_b) - 1)
    if example.kind == "complete_b":
        return LINKED_K_A * (np.sqrt(example.r_a) - 1), 0.0
    raise ValueError(f"Unexpected complete-harvest kind: {example.kind}")


def source_sink_candidate(example: Example) -> tuple[float, float]:
    if example.kind == "source_b_sink_a":
        return 0.0, LINKED_K_B * ((1 - example.m) * example.r_b - 1)
    if example.kind == "source_a_sink_b":
        return LINKED_K_A * ((1 - example.m) * example.r_a - 1), 0.0
    raise ValueError(f"Unexpected source-sink kind: {example.kind}")


def no_take_candidate(example: Example) -> tuple[float, float]:
    sqrt_r_a = np.sqrt(example.r_a)
    sqrt_r_b = np.sqrt(example.r_b)
    m = example.m

    def y_from_x(x_value: float) -> float:
        return float(
            np.sqrt(
                (1 - (1 - m) * x_value**2)
                / (1 - m + (2 * m - 1) * x_value**2)
            )
        )

    def x_from_y(y_value: float) -> float:
        return float(
            np.sqrt(
                (1 - (1 - m) * y_value**2)
                / (1 - m + (2 * m - 1) * y_value**2)
            )
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

        raise ValueError("No no-take root found")

    if example.kind == "no_take_a":
        def residual(x_value: float) -> float:
            y_value = y_from_x(x_value)
            population_a = LINKED_K_A * (sqrt_r_a - x_value) / x_value
            growth_a = LINKED_K_A * sqrt_r_a * (sqrt_r_a - x_value)
            growth_b = LINKED_K_B * sqrt_r_b * (sqrt_r_b - y_value)
            pre_a = (1 - m) * growth_a + m * growth_b
            return population_a - pre_a

        x = solve_root(residual, 1e-5, sqrt_r_a - 1e-5)
        y = y_from_x(x)
    elif example.kind == "no_take_b":
        def residual(y_value: float) -> float:
            x_value = x_from_y(y_value)
            population_b = LINKED_K_B * (sqrt_r_b - y_value) / y_value
            growth_a = LINKED_K_A * sqrt_r_a * (sqrt_r_a - x_value)
            growth_b = LINKED_K_B * sqrt_r_b * (sqrt_r_b - y_value)
            pre_b = m * growth_a + (1 - m) * growth_b
            return population_b - pre_b

        y = solve_root(residual, 1e-5, sqrt_r_b - 1e-5)
        x = x_from_y(y)
    else:
        raise ValueError(f"Unexpected no-take kind: {example.kind}")

    return (
        float(LINKED_K_A * (sqrt_r_a - x) / x),
        float(LINKED_K_B * (sqrt_r_b - y) / y),
    )


def candidate_population(example: Example) -> tuple[float, float]:
    if example.kind == "interior":
        return interior_candidate(example)
    if example.kind in {"complete_a", "complete_b"}:
        return complete_harvest_candidate(example)
    if example.kind in {"source_b_sink_a", "source_a_sink_b"}:
        return source_sink_candidate(example)
    if example.kind in {"no_take_a", "no_take_b"}:
        return no_take_candidate(example)
    raise ValueError(f"Unsupported example kind: {example.kind}")


def candidate_effort(example: Example) -> tuple[float, float]:
    if example.kind == "complete_a":
        e_a = 1.0
        e_b = 1 - 1 / ((1 - example.m) * np.sqrt(example.r_b))
        return e_a, float(e_b)
    if example.kind == "complete_b":
        e_a = 1 - 1 / ((1 - example.m) * np.sqrt(example.r_a))
        e_b = 1.0
        return float(e_a), e_b
    if example.kind == "source_b_sink_a":
        return 1.0, 0.0
    if example.kind == "source_a_sink_b":
        return 0.0, 1.0

    population_a, population_b = candidate_population(example)
    return recover_efforts(
        population_a,
        population_b,
        example.r_a,
        example.r_b,
        example.m,
    )


def parameter_text(example: Example) -> str:
    return (
        rf"$K_A={LINKED_K_A:g}$, $K_B={LINKED_K_B:g}$, "
        rf"$r_A={example.r_a:.3f}$, $r_B={example.r_b:.3f}$, "
        rf"$m={example.m:.3f}$"
    )


def grid_points_for(example: Example) -> int:
    return HIGH_RES_GRID_POINTS


def effort_yield_worker(args: tuple[float, float, float, float, float]) -> float:
    r_a, r_b, m, effort_a, effort_b = args
    _, _, yield_value = simulate_linked_equilibrium_and_yield(
        r_a=r_a,
        r_b=r_b,
        m=m,
        e_a=effort_a,
        e_b=effort_b,
        steps=SIMULATION_STEPS,
    )
    return float(yield_value)


def effort_yield_grid(example: Example) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_points = grid_points_for(example)
    efforts = np.linspace(0, 1, grid_points)
    effort_a_grid, effort_b_grid = np.meshgrid(efforts, efforts)
    jobs = [
        (
            example.r_a,
            example.r_b,
            example.m,
            float(effort_a),
            float(effort_b),
        )
        for effort_a, effort_b in zip(effort_a_grid.ravel(), effort_b_grid.ravel())
    ]

    if WORKER_COUNT == 1:
        values = [effort_yield_worker(job) for job in jobs]
    else:
        chunk_size = max(1, len(jobs) // (WORKER_COUNT * 8))
        with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
            values = list(executor.map(effort_yield_worker, jobs, chunksize=chunk_size))

    yield_grid = np.array(values, dtype=float).reshape(effort_a_grid.shape)
    return effort_a_grid, effort_b_grid, yield_grid


def effort_grid_result(example: Example) -> EffortGridResult:
    effort_a_grid, effort_b_grid, yields = effort_yield_grid(example)
    max_index = np.unravel_index(np.nanargmax(yields), yields.shape)
    max_effort_a = float(effort_a_grid[max_index])
    max_effort_b = float(effort_b_grid[max_index])
    max_population_a, max_population_b, max_yield = simulate_linked_equilibrium_and_yield(
        r_a=example.r_a,
        r_b=example.r_b,
        m=example.m,
        e_a=max_effort_a,
        e_b=max_effort_b,
        steps=SIMULATION_STEPS,
    )
    return EffortGridResult(
        effort_a_grid=effort_a_grid,
        effort_b_grid=effort_b_grid,
        yields=yields,
        max_effort_a=max_effort_a,
        max_effort_b=max_effort_b,
        max_population_a=float(max_population_a),
        max_population_b=float(max_population_b),
        max_yield=float(max_yield),
    )


def population_domain_grid(
    example: Example,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    upper_a = (1 - example.m) * LINKED_K_A * example.r_a + example.m * LINKED_K_B * example.r_b
    upper_b = example.m * LINKED_K_A * example.r_a + (1 - example.m) * LINKED_K_B * example.r_b
    candidate_a, candidate_b = candidate_population(example)
    upper_a = max(upper_a, candidate_a) * 1.05 + 0.05
    upper_b = max(upper_b, candidate_b) * 1.05 + 0.05

    grid_points = grid_points_for(example)
    population_a = np.linspace(0, upper_a, grid_points)
    population_b = np.linspace(0, upper_b, grid_points)
    population_a_grid, population_b_grid = np.meshgrid(population_a, population_b)

    pre_a, pre_b = pre_fishing_growth(
        population_a_grid,
        population_b_grid,
        example.r_a,
        example.r_b,
        example.m,
    )
    feasible = (population_a_grid <= pre_a + 1e-10) & (population_b_grid <= pre_b + 1e-10)
    yields = population_yield(
        population_a_grid,
        population_b_grid,
        example.r_a,
        example.r_b,
    )
    yields = np.where(feasible, yields, np.nan)
    yields = np.where(yields >= -1e-10, np.maximum(yields, 0), np.nan)
    return population_a_grid, population_b_grid, yields, feasible.astype(float)


def add_heatmap(
    ax,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    max_value: float | None = None,
):
    finite_values = z_grid[np.isfinite(z_grid)]
    if len(finite_values) == 0:
        raise ValueError("No finite yield values to plot")

    z_max = float(np.nanmax(finite_values)) if max_value is None else max_value
    levels = np.linspace(0, z_max, 31)
    heatmap = ax.contourf(
        x_grid,
        y_grid,
        z_grid,
        levels=levels,
        cmap="viridis",
    )
    contour_levels = np.linspace(0, z_max, 8)[1:]
    if np.nanmax(finite_values) > 0:
        contours = ax.contour(
            x_grid,
            y_grid,
            z_grid,
            levels=contour_levels,
            colors="white",
            linewidths=0.65,
            alpha=0.75,
        )
        ax.clabel(contours, inline=True, fontsize=6, fmt="%.2f")
    return heatmap


def plot_effort_domain(ax, example: Example, effort_grid: EffortGridResult) -> object:
    candidate_e_a, candidate_e_b = candidate_effort(example)
    heatmap = add_heatmap(
        ax,
        effort_grid.effort_a_grid,
        effort_grid.effort_b_grid,
        effort_grid.yields,
    )

    ax.scatter(
        [candidate_e_a],
        [candidate_e_b],
        color="red",
        edgecolor="white",
        linewidth=1.1,
        s=55,
        label="analytic candidate",
        zorder=4,
        clip_on=False,
    )
    ax.scatter(
        [effort_grid.max_effort_a],
        [effort_grid.max_effort_b],
        marker="x",
        color="black",
        linewidth=2,
        s=55,
        label="grid maximum",
        zorder=4,
        clip_on=False,
    )
    ax.set_xlabel(r"$e_A$")
    ax.set_ylabel(r"$e_B$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Effort domain")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="best", fontsize=7)
    return heatmap


def plot_population_domain(ax, example: Example, effort_grid: EffortGridResult) -> object:
    population_a_grid, population_b_grid, yields, feasible = population_domain_grid(example)
    candidate_a, candidate_b = candidate_population(example)
    heatmap = add_heatmap(ax, population_a_grid, population_b_grid, yields)

    ax.contour(
        population_a_grid,
        population_b_grid,
        feasible,
        levels=[0.5],
        colors="black",
        linewidths=1.2,
        linestyles="-",
    )
    ax.scatter(
        [candidate_a],
        [candidate_b],
        color="red",
        edgecolor="white",
        linewidth=1.1,
        s=55,
        label="analytic candidate",
        zorder=4,
        clip_on=False,
    )
    ax.scatter(
        [effort_grid.max_population_a],
        [effort_grid.max_population_b],
        marker="x",
        color="black",
        linewidth=2,
        s=55,
        label="grid maximum",
        zorder=4,
        clip_on=False,
    )
    ax.set_xlabel(r"$N_A$")
    ax.set_ylabel(r"$N_B$")
    finite = np.isfinite(yields)
    max_a = max(float(np.nanmax(population_a_grid[finite])), candidate_a)
    max_b = max(float(np.nanmax(population_b_grid[finite])), candidate_b)
    ax.set_xlim(0, max_a * 1.08 + 0.02)
    ax.set_ylim(0, max_b * 1.08 + 0.02)
    ax.set_title("Population domain")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="best", fontsize=7)
    return heatmap


def create_example_row(fig, axes, example: Example) -> object:
    effort_grid = effort_grid_result(example)
    heatmap = plot_effort_domain(axes[0], example, effort_grid)
    plot_population_domain(axes[1], example, effort_grid)
    candidate_a, candidate_b = candidate_population(example)
    candidate_e_a, candidate_e_b = candidate_effort(example)
    candidate_yield = population_yield(candidate_a, candidate_b, example.r_a, example.r_b)
    fig.suptitle(
        rf"{example.title}: {parameter_text(example)}"
        "\n"
        rf"candidate $(e_A,e_B)=({candidate_e_a:.3f},{candidate_e_b:.3f})$, "
        rf"$(N_A,N_B)=({candidate_a:.3f},{candidate_b:.3f})$, "
        rf"$Y={candidate_yield:.3f}$",
        fontsize=10,
    )
    return heatmap


def create_single_example_plot(example: Example, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    heatmap = create_example_row(fig, axes, example)
    colorbar = fig.colorbar(heatmap, ax=axes, shrink=0.88)
    colorbar.set_label("equilibrium yield")
    save_figure(fig, output_path, tight_layout=False)
    plt.close(fig)


def create_multi_example_plot(examples: tuple[Example, ...], output_path: Path) -> None:
    fig, axes = plt.subplots(
        len(examples),
        2,
        figsize=(10.5, 4.1 * len(examples)),
        constrained_layout=True,
    )
    if len(examples) == 1:
        axes = np.array([axes])

    heatmap = None
    for row, example in enumerate(examples):
        effort_grid = effort_grid_result(example)
        heatmap = plot_effort_domain(axes[row, 0], example, effort_grid)
        plot_population_domain(axes[row, 1], example, effort_grid)
        candidate_a, candidate_b = candidate_population(example)
        candidate_e_a, candidate_e_b = candidate_effort(example)
        candidate_yield = population_yield(candidate_a, candidate_b, example.r_a, example.r_b)
        axes[row, 0].set_ylabel(r"$e_B$" + "\n" + example.title)
        axes[row, 0].set_title(
            rf"Effort domain; {parameter_text(example)}"
            "\n"
            rf"candidate "
            rf"$({candidate_e_a:.3f},{candidate_e_b:.3f})$",
            fontsize=8,
        )
        axes[row, 1].set_title(
            rf"Population domain; {parameter_text(example)}"
            "\n"
            rf"candidate "
            rf"$({candidate_a:.3f},{candidate_b:.3f})$, $Y={candidate_yield:.3f}$",
            fontsize=8,
        )

    if heatmap is not None:
        colorbar = fig.colorbar(heatmap, ax=axes, shrink=0.92)
        colorbar.set_label("equilibrium yield")

    save_figure(fig, output_path, tight_layout=False)
    plt.close(fig)


def main() -> None:
    create_single_example_plot(
        Example(
            title="Interior candidate",
            r_a=1.8,
            r_b=1.2,
            m=0.2,
            kind="interior",
        ),
        PLOTS_DIR / "population_domain_view_interior.pdf",
    )
    create_multi_example_plot(
        (
            Example(
                title="Complete harvest in A",
                r_a=0.9274,
                r_b=4.3917,
                m=0.1784,
                kind="complete_a",
            ),
            Example(
                title="Complete harvest in B",
                r_a=3.4753,
                r_b=0.4583,
                m=0.2171,
                kind="complete_b",
            ),
        ),
        PLOTS_DIR / "population_domain_view_complete_harvest.pdf",
    )
    create_multi_example_plot(
        (
            Example(
                title="Protect A, harvest B",
                r_a=4.0,
                r_b=0.8,
                m=0.5,
                kind="source_a_sink_b",
            ),
            Example(
                title="Protect B, harvest A",
                r_a=0.8,
                r_b=1 / (1 - 0.3) ** 2,
                m=0.3,
                kind="source_b_sink_a",
            ),
        ),
        PLOTS_DIR / "population_domain_view_source_endpoint.pdf",
    )
    create_multi_example_plot(
        (
            Example(
                title="No take in A",
                r_a=1.8685,
                r_b=1.0365,
                m=0.7586,
                kind="no_take_a",
            ),
            Example(
                title="No take in B",
                r_a=0.8367,
                r_b=2.7031,
                m=0.9817,
                kind="no_take_b",
            ),
        ),
        PLOTS_DIR / "population_domain_view_no_take.pdf",
    )


if __name__ == "__main__":
    main()
