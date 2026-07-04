from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plot_generators.common import linked_spectral_radius


EXTINCTION = 0
INTERIOR = 1
NO_TAKE_A = 2
NO_TAKE_B = 3
COMPLETE_A = 4
COMPLETE_B = 5
PROTECT_A_HARVEST_B = 6
PROTECT_B_HARVEST_A = 7


class BlackBackedLine:
    def __init__(self, *, linestyle: str, label: str) -> None:
        self.linestyle = linestyle
        self.label = label

    def get_label(self) -> str:
        return self.label


class BlackBackedLineHandler(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        background = plt.Rectangle(
            (xdescent, ydescent + height * 0.18),
            width,
            height * 0.64,
            facecolor="black",
            edgecolor="black",
            transform=trans,
        )
        line = Line2D(
            [xdescent + width * 0.12, xdescent + width * 0.88],
            [ydescent + height * 0.5, ydescent + height * 0.5],
            color="white",
            linestyle=orig_handle.linestyle,
            linewidth=2.0,
            transform=trans,
        )
        return [background, line]


@dataclass(frozen=True)
class RowTask:
    row: int
    r_b: float
    r_a_values: np.ndarray
    m: float
    k_a: float
    k_b: float
    effort_grid_size: int
    local_effort_grid_size: int
    steps: int
    local_steps: int
    yield_tolerance: float
    classification_tolerance: float


def classify_effort(
    effort_a: float,
    effort_b: float,
    *,
    edge_tolerance: float,
) -> int:
    effort_a_zero = effort_a <= edge_tolerance
    effort_b_zero = effort_b <= edge_tolerance
    effort_a_one = effort_a >= 1 - edge_tolerance
    effort_b_one = effort_b >= 1 - edge_tolerance

    if effort_a_zero and effort_b_one:
        return PROTECT_A_HARVEST_B
    if effort_a_one and effort_b_zero:
        return PROTECT_B_HARVEST_A
    if effort_a_one:
        return COMPLETE_A
    if effort_b_one:
        return COMPLETE_B
    if effort_a_zero:
        return NO_TAKE_A
    if effort_b_zero:
        return NO_TAKE_B
    return INTERIOR


def simulate_equilibrium_yield(
    effort_a: float,
    effort_b: float,
    *,
    r_a: float,
    r_b: float,
    m: float,
    k_a: float,
    k_b: float,
    steps: int,
) -> float:
    population_a = 0.5
    population_b = 0.5
    yield_value = 0.0

    for _ in range(steps):
        growth_a = k_a * r_a * population_a / (k_a + population_a)
        growth_b = k_b * r_b * population_b / (k_b + population_b)
        pre_fishing_a = (1 - m) * growth_a + m * growth_b
        pre_fishing_b = m * growth_a + (1 - m) * growth_b
        yield_value = effort_a * pre_fishing_a + effort_b * pre_fishing_b
        population_a = (1 - effort_a) * pre_fishing_a
        population_b = (1 - effort_b) * pre_fishing_b

    return float(yield_value)


def best_effort_on_grid(
    r_a: float,
    r_b: float,
    *,
    m: float,
    k_a: float,
    k_b: float,
    effort_a_min: float,
    effort_a_max: float,
    effort_b_min: float,
    effort_b_max: float,
    effort_grid_size: int,
    steps: int,
) -> tuple[float, float, float, float, float]:
    efforts_a = np.linspace(effort_a_min, effort_a_max, effort_grid_size)
    efforts_b = np.linspace(effort_b_min, effort_b_max, effort_grid_size)
    effort_a, effort_b = np.meshgrid(efforts_a, efforts_b)
    population_a = np.full_like(effort_a, 0.5, dtype=float)
    population_b = np.full_like(effort_b, 0.5, dtype=float)
    yield_values = np.zeros_like(effort_a, dtype=float)

    for _ in range(steps):
        growth_a = k_a * r_a * population_a / (k_a + population_a)
        growth_b = k_b * r_b * population_b / (k_b + population_b)
        pre_fishing_a = (1 - m) * growth_a + m * growth_b
        pre_fishing_b = m * growth_a + (1 - m) * growth_b
        yield_values = effort_a * pre_fishing_a + effort_b * pre_fishing_b
        population_a = (1 - effort_a) * pre_fishing_a
        population_b = (1 - effort_b) * pre_fishing_b

    best_yield = float(np.nanmax(yield_values))
    row, column = np.unravel_index(np.nanargmax(yield_values), yield_values.shape)
    best_effort_a = float(effort_a[row, column])
    best_effort_b = float(effort_b[row, column])
    effort_a_step = (effort_a_max - effort_a_min) / max(1, effort_grid_size - 1)
    effort_b_step = (effort_b_max - effort_b_min) / max(1, effort_grid_size - 1)
    return best_effort_a, best_effort_b, best_yield, effort_a_step, effort_b_step


def local_neighborhood_effort_search(
    r_a: float,
    r_b: float,
    *,
    m: float,
    k_a: float,
    k_b: float,
    effort_grid_size: int,
    local_effort_grid_size: int,
    steps: int,
    local_steps: int,
) -> tuple[float, float, float]:
    (
        coarse_effort_a,
        coarse_effort_b,
        _coarse_yield,
        effort_a_step,
        effort_b_step,
    ) = best_effort_on_grid(
        r_a,
        r_b,
        m=m,
        k_a=k_a,
        k_b=k_b,
        effort_a_min=0.0,
        effort_a_max=1.0,
        effort_b_min=0.0,
        effort_b_max=1.0,
        effort_grid_size=effort_grid_size,
        steps=steps,
    )

    effort_a_min = max(0.0, coarse_effort_a - effort_a_step)
    effort_a_max = min(1.0, coarse_effort_a + effort_a_step)
    effort_b_min = max(0.0, coarse_effort_b - effort_b_step)
    effort_b_max = min(1.0, coarse_effort_b + effort_b_step)

    best_effort_a, best_effort_b, best_yield, _step_a, _step_b = best_effort_on_grid(
        r_a,
        r_b,
        m=m,
        k_a=k_a,
        k_b=k_b,
        effort_a_min=effort_a_min,
        effort_a_max=effort_a_max,
        effort_b_min=effort_b_min,
        effort_b_max=effort_b_max,
        effort_grid_size=local_effort_grid_size,
        steps=local_steps,
    )

    return best_effort_a, best_effort_b, best_yield


def numeric_zone(
    r_a: float,
    r_b: float,
    *,
    m: float,
    k_a: float,
    k_b: float,
    effort_grid_size: int,
    local_effort_grid_size: int,
    steps: int,
    local_steps: int,
    yield_tolerance: float,
    classification_tolerance: float,
) -> int:
    best_effort_a, best_effort_b, best_yield = local_neighborhood_effort_search(
        r_a,
        r_b,
        m=m,
        k_a=k_a,
        k_b=k_b,
        effort_grid_size=effort_grid_size,
        local_effort_grid_size=local_effort_grid_size,
        steps=steps,
        local_steps=local_steps,
    )
    if best_yield <= yield_tolerance:
        return EXTINCTION

    return classify_effort(
        best_effort_a,
        best_effort_b,
        edge_tolerance=classification_tolerance,
    )


def classify_row(task: RowTask) -> tuple[int, np.ndarray]:
    zones = np.array(
        [
            numeric_zone(
                float(r_a),
                task.r_b,
                m=task.m,
                k_a=task.k_a,
                k_b=task.k_b,
                effort_grid_size=task.effort_grid_size,
                local_effort_grid_size=task.local_effort_grid_size,
                steps=task.steps,
                local_steps=task.local_steps,
                yield_tolerance=task.yield_tolerance,
                classification_tolerance=task.classification_tolerance,
            )
            for r_a in task.r_a_values
        ],
        dtype=int,
    )
    return task.row, zones


def create_numeric_zone_grid(
    *,
    grid_size: int,
    r_min: float,
    r_max: float,
    m: float,
    k_a: float,
    k_b: float,
    effort_grid_size: int,
    local_effort_grid_size: int,
    steps: int,
    local_steps: int,
    processes: int,
    yield_tolerance: float,
    classification_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_a_values = np.linspace(r_min, r_max, grid_size)
    r_b_values = np.linspace(r_min, r_max, grid_size)
    tasks = [
        RowTask(
            row=row,
            r_b=float(r_b),
            r_a_values=r_a_values,
            m=m,
            k_a=k_a,
            k_b=k_b,
            effort_grid_size=effort_grid_size,
            local_effort_grid_size=local_effort_grid_size,
            steps=steps,
            local_steps=local_steps,
            yield_tolerance=yield_tolerance,
            classification_tolerance=classification_tolerance,
        )
        for row, r_b in enumerate(r_b_values)
    ]

    if processes == 1:
        results = [classify_row(task) for task in tasks]
    else:
        with Pool(processes=processes) as pool:
            results = pool.map(classify_row, tasks)

    zones = np.zeros((grid_size, grid_size), dtype=int)
    for row, row_zones in results:
        zones[row, :] = row_zones

    return r_a_values, r_b_values, zones


def interior_condition_values(
    r_a_values: np.ndarray,
    r_b_values: np.ndarray,
    *,
    m: float,
    k_a: float,
    k_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    r_a, r_b = np.meshgrid(r_a_values, r_b_values)
    sqrt_r_a = np.sqrt(r_a)
    sqrt_r_b = np.sqrt(r_b)
    middle = m * (
        k_a * sqrt_r_a * (sqrt_r_a - 1)
        - k_b * sqrt_r_b * (sqrt_r_b - 1)
    )
    lower = -k_b * (sqrt_r_b - 1) ** 2
    upper = k_a * (sqrt_r_a - 1) ** 2
    mask = (r_a > 1) & (r_b > 1)
    return (
        np.where(mask, middle - lower, np.nan),
        np.where(mask, middle - upper, np.nan),
    )


def protect_source_sink_boundary(source_r: np.ndarray, m: float) -> np.ndarray:
    retention = 1 - m
    return (
        retention * (retention * source_r - 1)
        / (retention**3 * source_r + 2 * m - 1)
    )


def plot_condition_overlays(
    ax,
    r_a_values: np.ndarray,
    r_b_values: np.ndarray,
    *,
    m: float,
    k_a: float,
    k_b: float,
) -> list[Line2D]:
    r_a, r_b = np.meshgrid(r_a_values, r_b_values)

    spectral_radius = linked_spectral_radius(r_a, r_b, m)
    ax.contour(
        r_a,
        r_b,
        spectral_radius,
        levels=[1],
        colors="black",
        linewidths=2.4,
    )

    lower_gap, upper_gap = interior_condition_values(
        r_a_values,
        r_b_values,
        m=m,
        k_a=k_a,
        k_b=k_b,
    )
    ax.contour(
        r_a,
        r_b,
        lower_gap,
        levels=[0],
        colors="white",
        linewidths=2.0,
        linestyles="--",
    )
    ax.contour(
        r_a,
        r_b,
        upper_gap,
        levels=[0],
        colors="white",
        linewidths=2.0,
        linestyles=":",
    )

    retention = 1 - m
    complete_threshold = 1 / retention**2 if retention > 0 else np.inf
    r_min = float(r_a_values[0])
    r_max = float(r_a_values[-1])
    if r_min <= complete_threshold <= r_max:
        ax.plot(
            [r_min, min(1, r_max)],
            [complete_threshold, complete_threshold],
            color="#ffff00",
            linestyle="--",
            linewidth=2.0,
        )
        ax.plot(
            [complete_threshold, complete_threshold],
            [r_min, min(1, r_max)],
            color="#ffff00",
            linestyle="--",
            linewidth=2.0,
        )
    if r_min <= 1 <= r_max:
        ax.plot(
            [1, 1],
            [max(r_min, complete_threshold), r_max],
            color="#ffff00",
            linestyle="--",
            linewidth=2.0,
        )
        ax.plot(
            [max(r_min, complete_threshold), r_max],
            [1, 1],
            color="#ffff00",
            linestyle="--",
            linewidth=2.0,
        )

    if retention > 0:
        source_min = max(r_min, 1 / retention)
        source_max = min(r_max, 1 / retention**2)
        if source_min < source_max:
            source = np.linspace(source_min, source_max, 400)
            sink_limit = protect_source_sink_boundary(source, m)
            valid = np.isfinite(sink_limit) & (r_min <= sink_limit) & (sink_limit <= r_max)
            ax.plot(
                sink_limit[valid],
                source[valid],
                color="#ff00a8",
                linestyle="-.",
                linewidth=2.8,
                zorder=5,
            )
            ax.plot(
                source[valid],
                sink_limit[valid],
                color="#00d5ff",
                linestyle="-.",
                linewidth=2.8,
                zorder=5,
            )

    return [
        Line2D([0], [0], color="black", linewidth=2.4, label=r"$\rho(M)=1$"),
        Line2D(
            [0],
            [0],
            color="#ffff00",
            linestyle="--",
            linewidth=2.0,
            label="complete harvest in A/B boundary",
        ),
        Line2D(
            [0],
            [0],
            color="#00d5ff",
            linestyle="-.",
            linewidth=2.8,
            label="protect A, harvest B boundary",
        ),
        Line2D(
            [0],
            [0],
            color="#ff00a8",
            linestyle="-.",
            linewidth=2.8,
            label="protect B, harvest A boundary",
        ),
    ]


def interior_boundary_handles() -> list[Line2D]:
    return [
        BlackBackedLine(
            linestyle="--",
            label="interior boundary, lower inequality",
        ),
        BlackBackedLine(
            linestyle=":",
            label="interior boundary, upper inequality",
        ),
    ]


def plot_zone_map(
    r_a_values: np.ndarray,
    r_b_values: np.ndarray,
    zones: np.ndarray,
    *,
    m: float,
    k_a: float,
    k_b: float,
    output: Path,
) -> None:
    colors = [
        "#d9d9d9",
        "#4c78a8",
        "#54a24b",
        "#1b9e77",
        "#f58518",
        "#f2cf5b",
        "#762a83",
        "#b2182b",
    ]
    labels = [
        "extinction / zero sustainable yield",
        "interior",
        "no take in A",
        "no take in B",
        "complete harvest in A",
        "complete harvest in B",
        "protect A, harvest B endpoint",
        "protect B, harvest A endpoint",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap.N)

    fig, ax = plt.subplots(figsize=(8.2, 7.8))
    fig.subplots_adjust(bottom=0.32)
    ax.imshow(
        zones,
        origin="lower",
        extent=(r_a_values[0], r_a_values[-1], r_b_values[0], r_b_values[-1]),
        interpolation="nearest",
        aspect="equal",
        cmap=cmap,
        norm=norm,
    )
    line_handles = plot_condition_overlays(
        ax,
        r_a_values,
        r_b_values,
        m=m,
        k_a=k_a,
        k_b=k_b,
    )

    ax.set_xlabel(r"$r_A$")
    ax.set_ylabel(r"$r_B$")
    ax.set_title(
        rf"Pure numeric yield maximizer, $m={m:g}$, "
        rf"$K_A={k_a:g}$, $K_B={k_b:g}$"
    )
    ax.grid(True, color="white", linewidth=0.4, alpha=0.35)

    patch_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.4, label=label)
        for color, label in zip(colors, labels)
    ]
    fig.legend(
        handles=patch_handles + line_handles + interior_boundary_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        fontsize=8,
        handler_map={BlackBackedLine: BlackBackedLineHandler()},
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=float, default=0.45)
    parser.add_argument("--k-a", type=float, default=2.0)
    parser.add_argument("--k-b", type=float, default=1.0)
    parser.add_argument("--r-min", type=float, default=0.0)
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--grid-size", type=int, default=51)
    parser.add_argument("--effort-grid-size", type=int, default=40)
    parser.add_argument("--local-effort-grid-size", type=int, default=100)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--local-steps", type=int, default=300)
    parser.add_argument("--yield-tolerance", type=float, default=1e-8)
    parser.add_argument("--classification-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--processes",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/numeric_yield_zone_map_ka2_kb1_m0_45_lowres.pdf"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    r_a_values, r_b_values, zones = create_numeric_zone_grid(
        grid_size=args.grid_size,
        r_min=args.r_min,
        r_max=args.r_max,
        m=args.m,
        k_a=args.k_a,
        k_b=args.k_b,
        effort_grid_size=args.effort_grid_size,
        local_effort_grid_size=args.local_effort_grid_size,
        steps=args.steps,
        local_steps=args.local_steps,
        processes=args.processes,
        yield_tolerance=args.yield_tolerance,
        classification_tolerance=args.classification_tolerance,
    )
    plot_zone_map(
        r_a_values,
        r_b_values,
        zones,
        m=args.m,
        k_a=args.k_a,
        k_b=args.k_b,
        output=args.output,
    )
    counts = np.bincount(zones.ravel(), minlength=8)
    print(f"grid size: {args.grid_size} x {args.grid_size}")
    print(f"coarse effort grid: {args.effort_grid_size} x {args.effort_grid_size}")
    print(f"coarse steps: {args.steps}")
    print(f"local effort grid: {args.local_effort_grid_size} x {args.local_effort_grid_size}")
    print(f"local steps: {args.local_steps}")
    print(f"processes: {args.processes}")
    print(f"classification tolerance: {args.classification_tolerance}")
    print(f"extinction / zero yield: {counts[EXTINCTION]}")
    print(f"interior: {counts[INTERIOR]}")
    print(f"no take in A: {counts[NO_TAKE_A]}")
    print(f"no take in B: {counts[NO_TAKE_B]}")
    print(f"complete harvest in A: {counts[COMPLETE_A]}")
    print(f"complete harvest in B: {counts[COMPLETE_B]}")
    print(f"protect A, harvest B endpoint: {counts[PROTECT_A_HARVEST_B]}")
    print(f"protect B, harvest A endpoint: {counts[PROTECT_B_HARVEST_A]}")
    print(f"saved {args.output}")
    print(f"saved {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
