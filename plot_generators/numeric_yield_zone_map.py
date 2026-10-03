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
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plot_generators.common import linked_spectral_radius
from plot_generators.equilibrium_simulation import equilibrium_yields


EXTINCTION = 0
INTERIOR = 1
NO_TAKE_A = 2
NO_TAKE_B = 3
COMPLETE_A = 4
COMPLETE_B = 5
PROTECT_A_HARVEST_B = 6
PROTECT_B_HARVEST_A = 7
ALGORITHM = "bounded_equilibrium_iteration_v1"


CURVE_COLORS = {
    "extinction": "#243847",
    "complete": "#936322",
    "source": "#784c77",
    "interior": "#28475d",
}


@dataclass(frozen=True)
class RowTask:
    row: int
    r_b: float
    r_a_values: np.ndarray
    parameters: dict


def classify_effort(effort_a: float, effort_b: float, *, edge_tolerance: float) -> int:
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


def simulate_equilibrium_yield(effort_a: float, effort_b: float, **parameters) -> float:
    """Evaluate the same population update until the catch bounds agree."""
    return float(equilibrium_yields(effort_a, effort_b, **parameters))


def best_effort_on_grid(
    r_a: float, r_b: float, *, effort_a_min: float, effort_a_max: float,
    effort_b_min: float, effort_b_max: float, effort_grid_size: int,
    previous_best: tuple[float, float] | None = None, **parameters,
) -> tuple[float, float, float, float, float]:
    efforts_a = np.linspace(effort_a_min, effort_a_max, effort_grid_size)
    efforts_b = np.linspace(effort_b_min, effort_b_max, effort_grid_size)
    # Preserve the incumbent and exact edge efforts in every refinement.
    if previous_best is not None:
        efforts_a = np.unique(np.append(efforts_a, previous_best[0]))
        efforts_b = np.unique(np.append(efforts_b, previous_best[1]))
    effort_a, effort_b = np.meshgrid(efforts_a, efforts_b)
    yields = equilibrium_yields(effort_a, effort_b, r_a=r_a, r_b=r_b, **parameters)
    row, column = np.unravel_index(np.argmax(yields), yields.shape)
    step_a = (effort_a_max - effort_a_min) / (effort_grid_size - 1)
    step_b = (effort_b_max - effort_b_min) / (effort_grid_size - 1)
    return (float(effort_a[row, column]), float(effort_b[row, column]),
            float(yields[row, column]), step_a, step_b)


def local_neighborhood_effort_search(
    r_a: float, r_b: float, *, m: float, k_a: float, k_b: float,
    effort_grid_size: int = 40, local_effort_grid_size: int = 9,
    effort_tolerance: float = 1e-5, yield_atol: float = 1e-12,
    yield_rtol: float = 1e-9, max_iterations: int = 200,
    max_search_iterations: int = 200,
) -> tuple[float, float, float]:
    """Refine a coarse effort search after clearing artificial window edges."""
    if effort_grid_size < 3 or local_effort_grid_size < 5 or effort_tolerance <= 0:
        raise ValueError("Use at least 3 coarse / 5 local points and a positive effort tolerance.")
    if not isinstance(max_search_iterations, (int, np.integer)) or max_search_iterations < 1:
        raise ValueError("max_search_iterations must be a positive integer")
    parameters = dict(m=m, k_a=k_a, k_b=k_b, yield_atol=yield_atol,
                      yield_rtol=yield_rtol, max_iterations=max_iterations)
    ea, eb, best_yield, da, db = best_effort_on_grid(
        r_a, r_b, effort_a_min=0, effort_a_max=1,
        effort_b_min=0, effort_b_max=1,
        effort_grid_size=effort_grid_size, **parameters,
    )
    edge_slack = 32 * np.finfo(float).eps

    def on_artificial_edge(effort, lower, upper):
        return ((lower > 0 and effort <= lower + edge_slack)
                or (upper < 1 and effort >= upper - edge_slack))

    for _ in range(max_search_iterations):
        if max(da, db) <= effort_tolerance:
            return ea, eb, best_yield
        a_min, a_max = max(0, ea - da), min(1, ea + da)
        b_min, b_max = max(0, eb - db), min(1, eb + db)
        ea, eb, best_yield, next_da, next_db = best_effort_on_grid(
            r_a, r_b, effort_a_min=a_min, effort_a_max=a_max,
            effort_b_min=b_min, effort_b_max=b_max,
            effort_grid_size=local_effort_grid_size, previous_best=(ea, eb),
            **parameters,
        )
        # An optimum on an artificial edge can lie beyond this window. Move
        # the window at its current resolution before reducing its radius.
        # The physical effort boundaries 0 and 1 remain legitimate optima.
        if not (on_artificial_edge(ea, a_min, a_max)
                or on_artificial_edge(eb, b_min, b_max)):
            da, db = next_da, next_db
    if max(da, db) <= effort_tolerance:
        return ea, eb, best_yield
    raise RuntimeError(
        f"effort search did not converge after {max_search_iterations} refinements "
        f"(remaining effort spacing {max(da, db):.3g})"
    )


def numeric_zone(
    r_a: float, r_b: float, *, m: float, k_a: float, k_b: float,
    effort_grid_size: int = 40, local_effort_grid_size: int = 9,
    effort_tolerance: float = 1e-5, yield_tolerance: float = 1e-10,
    classification_tolerance: float = 1e-4, yield_atol: float = 1e-12,
    yield_rtol: float = 1e-9, max_iterations: int = 200,
) -> int:
    ea, eb, best_yield = local_neighborhood_effort_search(
        r_a, r_b, m=m, k_a=k_a, k_b=k_b, effort_grid_size=effort_grid_size,
        local_effort_grid_size=local_effort_grid_size, effort_tolerance=effort_tolerance,
        yield_atol=yield_atol, yield_rtol=yield_rtol, max_iterations=max_iterations,
    )
    if best_yield <= yield_tolerance:
        return EXTINCTION
    return classify_effort(ea, eb, edge_tolerance=classification_tolerance)


def classify_row(task: RowTask) -> tuple[int, np.ndarray]:
    zones = np.array([numeric_zone(float(r_a), task.r_b, **task.parameters)
                      for r_a in task.r_a_values], dtype=int)
    return task.row, zones


def create_numeric_zone_grid(
    *, grid_size: int, r_min: float, r_max: float, processes: int, **parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_size < 2 or not 0 <= r_min < r_max:
        raise ValueError("Use at least two grid points and 0 <= r_min < r_max.")
    r_a_values = np.linspace(r_min, r_max, grid_size)
    r_b_values = np.linspace(r_min, r_max, grid_size)
    tasks = [RowTask(row, float(r_b), r_a_values, parameters)
             for row, r_b in enumerate(r_b_values)]
    zones = np.zeros((grid_size, grid_size), dtype=int)

    def collect(results):
        for completed, (row, row_zones) in enumerate(results, 1):
            zones[row, :] = row_zones
            if completed % 10 == 0 or completed == grid_size:
                print(f"computed {completed}/{grid_size} rows", flush=True)

    if processes == 1:
        collect(map(classify_row, tasks))
    else:
        with Pool(processes=processes) as pool:
            collect(pool.imap_unordered(classify_row, tasks))
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
) -> dict[str, tuple[float, float]]:
    r_a, r_b = np.meshgrid(r_a_values, r_b_values)
    markers = {}

    def mark_contour(contour, label, target):
        segments = [segment for segment in contour.allsegs[0] if len(segment)]
        if segments:
            points = np.concatenate(segments)
            nearest = points[np.argmin(np.sum((points - target) ** 2, axis=1))]
            markers[label] = tuple(nearest)

    spectral_radius = linked_spectral_radius(r_a, r_b, m)
    extinction_curve = ax.contour(
        r_a,
        r_b,
        spectral_radius,
        levels=[1],
        colors=CURVE_COLORS["extinction"],
        linewidths=2.0,
    )
    mark_contour(extinction_curve, "1", (0.52, 1.45))

    lower_gap, upper_gap = interior_condition_values(
        r_a_values,
        r_b_values,
        m=m,
        k_a=k_a,
        k_b=k_b,
    )
    lower_curve = ax.contour(
        r_a,
        r_b,
        lower_gap,
        levels=[0],
        colors=CURVE_COLORS["interior"],
        linewidths=1.8,
        linestyles="--",
    )
    upper_curve = ax.contour(
        r_a,
        r_b,
        upper_gap,
        levels=[0],
        colors=CURVE_COLORS["interior"],
        linewidths=1.8,
        linestyles=":",
    )
    mark_contour(lower_curve, "4B", (1.18, 2.22))
    mark_contour(upper_curve, "4A", (2.25, 1.7))

    retention = 1 - m
    complete_threshold = 1 / retention**2 if retention > 0 else np.inf
    r_min = float(r_a_values[0])
    r_max = float(r_a_values[-1])
    if r_min <= complete_threshold <= r_max:
        if r_min <= min(1, r_max):
            midpoint = (r_min + min(1, r_max)) / 2
            markers["2-left"] = (midpoint, complete_threshold)
            markers["2-bottom"] = (complete_threshold, midpoint)
        ax.plot(
            [r_min, min(1, r_max)],
            [complete_threshold, complete_threshold],
            color=CURVE_COLORS["complete"],
            linestyle="--",
            linewidth=2.0,
        )
        ax.plot(
            [complete_threshold, complete_threshold],
            [r_min, min(1, r_max)],
            color=CURVE_COLORS["complete"],
            linestyle="--",
            linewidth=2.0,
        )
    if r_min <= 1 <= r_max:
        ax.plot(
            [1, 1],
            [max(r_min, complete_threshold), r_max],
            color=CURVE_COLORS["complete"],
            linestyle="--",
            linewidth=2.0,
        )
        ax.plot(
            [max(r_min, complete_threshold), r_max],
            [1, 1],
            color=CURVE_COLORS["complete"],
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
                color=CURVE_COLORS["source"],
                linestyle="-.",
                linewidth=2.0,
                zorder=5,
            )
            ax.plot(
                source[valid],
                sink_limit[valid],
                color=CURVE_COLORS["source"],
                linestyle="-.",
                linewidth=2.0,
                zorder=5,
            )
            if np.any(valid):
                middle_index = len(source[valid]) // 2
                s = float(source[valid][middle_index])
                h = float(sink_limit[valid][middle_index])
                markers["3-left"] = (h, s)
                markers["3-bottom"] = (s, h)
    return markers


def add_equation_key(ax, *, k_a: float, k_b: float, transform=None) -> None:
    """Explain the analytic overlays in unused space in the interior region."""
    transform = ax.transAxes if transform is None else transform
    card = FancyBboxPatch(
        (0.285, 0.414), 0.69, 0.563,
        boxstyle="round,pad=0.012,rounding_size=0.013",
        transform=transform, facecolor="white", edgecolor="#c1cbd3",
        linewidth=0.8, alpha=0.96, zorder=8,
    )
    ax.add_patch(card)

    def text(x, y, value, *, size=14, weight="normal", color="#243847"):
        ax.text(x, y, value, transform=transform, fontsize=size,
                fontweight=weight, color=color, ha="left", va="center", zorder=9)

    def heading(y, number, title, color, linestyle):
        ax.plot([0.308, 0.351], [y, y], transform=transform,
                color=color, linestyle=linestyle, linewidth=2, zorder=9)
        text(0.367, y, f"{number}  {title}", size=13.5, weight="bold")

    text(0.308, 0.950, "BOUNDARY EQUATIONS", size=14, weight="bold")
    text(0.308, 0.913, r"$x=\sqrt{r_A},\quad y=\sqrt{r_B},\quad q=1-m$", size=14)
    heading(0.868, "1", "Extinction", CURVE_COLORS["extinction"], "-")
    text(0.308, 0.830, r"$q(r_A+r_B)-(1-2m)r_A r_B=1$")
    heading(0.782, "2", "Complete harvest", CURVE_COLORS["complete"], "--")
    text(0.308, 0.744, r"$r_A=1\ \mathrm{or}\ r_B=q^{-2}\quad (A\leftrightarrow B)$")
    heading(0.696, "3", "Protect source, harvest sink", CURVE_COLORS["source"], "-.")
    text(0.308, 0.660, r"$r_B=H(r_A)\ \mathrm{or}\ r_A=H(r_B)$")
    text(0.308, 0.612, r"$H(s)=\dfrac{q(qs-1)}{q^3s+2m-1}$")
    text(0.308, 0.557, "4  Interior / no-take", size=13.5, weight="bold")
    coefficient_a = "" if k_a == 1 else f"{k_a:g}"
    coefficient_b = "" if k_b == 1 else f"{k_b:g}"
    for y, suffix, style, equation in [
        (0.515, "A", ":", rf"${coefficient_a}(x-1)(1-qx)={coefficient_b}m\,y(y-1)$"),
        (0.472, "B", "--", rf"${coefficient_b}(y-1)(1-qy)={coefficient_a}m\,x(x-1)$"),
    ]:
        ax.plot([0.308, 0.351], [y, y], transform=transform,
                color=CURVE_COLORS["interior"], linestyle=style, linewidth=2, zorder=9)
        text(0.367, y, suffix + ": " + equation, size=13)
    text(0.308, 0.431, "Equations apply to the plotted segments.", size=11,
         color="#596975")


def add_map_labels(ax, r_a_values, r_b_values, zones, markers, *, inset_key: bool) -> None:
    """Prefer hand-tuned positions, falling back to the actual region interiors."""
    ra, rb = np.meshgrid(r_a_values, r_b_values)
    card_mask = equation_card_mask(r_a_values, r_b_values) if inset_key else np.zeros_like(zones, dtype=bool)
    for zone, x, y, label, size in [
        (EXTINCTION, 0.49, 0.52, "Extinction\nzero yield", 13),
        (COMPLETE_A, 0.49, 4.22, "Complete\nharvest\nin A", 14),
        (COMPLETE_B, 4.18, 0.49, "Complete harvest\nin B", 14),
        (PROTECT_B_HARVEST_A, 0.35, 2.87, "Protect B\nharvest A", 12),
        (PROTECT_A_HARVEST_B, 2.58, 0.27, "Protect A\nharvest B", 12),
        (NO_TAKE_B, 0.78, 1.84, "No take\nin B", 13),
        (NO_TAKE_A, 2.05, 1.16, "No take in A", 14),
        (INTERIOR, 3.94, 1.69, "Interior\n" + r"$0<e_A,e_B<1$", 15),
    ]:
        region = (zones == zone) & ~card_mask
        if not np.any(region):
            continue
        row, col = np.argmin(abs(r_b_values - y)), np.argmin(abs(r_a_values - x))
        if not (region[row, col] and r_a_values[0] <= x <= r_a_values[-1]
                and r_b_values[0] <= y <= r_b_values[-1]):
            distances = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
            row, col = np.unravel_index(np.argmax(distances), region.shape)
            x, y = ra[row, col], rb[row, col]
        ax.text(x, y, label, color="#243847", fontsize=size,
                ha="center", va="center", linespacing=1.25,
                fontweight="bold", zorder=7)

    families = {"1": "extinction", "2": "complete", "3": "source", "4": "interior"}
    for key, (x, y) in markers.items():
        label = key.split("-")[0]
        color = CURVE_COLORS[families[label[0]]]
        ax.text(x, y, label, fontsize=11, color=color, fontweight="bold",
                ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor=color, linewidth=1.1))


def equation_card_mask(r_a_values, r_b_values):
    x, y = np.meshgrid(
        (r_a_values - r_a_values[0]) / (r_a_values[-1] - r_a_values[0]),
        (r_b_values - r_b_values[0]) / (r_b_values[-1] - r_b_values[0]),
    )
    return (x >= 0.27) & (x <= 0.99) & (y >= 0.40) & (y <= 0.99)


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
        "#e6e8eb", "#d9e6f2", "#9dbfaa", "#79b6b1",
        "#f1bf79", "#ecd88c", "#b19ac6", "#d18b97",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap.N)

    fig = plt.figure(figsize=(8.8, 9.0))
    ax = fig.add_axes((0.105, 0.08, 0.85, 0.83))
    # The simulated coordinates are cell centers, not the image's outer edges.
    ax.pcolormesh(r_a_values, r_b_values, zones, shading="nearest",
                  cmap=cmap, norm=norm, rasterized=True)
    ax.set(xlim=(r_a_values[0], r_a_values[-1]),
           ylim=(r_b_values[0], r_b_values[-1]), aspect="equal")
    markers = plot_condition_overlays(
        ax,
        r_a_values,
        r_b_values,
        m=m,
        k_a=k_a,
        k_b=k_b,
    )

    ax.set_xlabel(r"$r_A$", fontsize=18, labelpad=8)
    ax.set_ylabel(r"$r_B$", fontsize=18, labelpad=8)
    ax.tick_params(labelsize=14, length=4, color="#596975")
    ax.grid(True, color="#314553", linewidth=0.4, alpha=0.12)
    for spine in ax.spines.values():
        spine.set_color("#596975")
        spine.set_linewidth(0.8)
    fig.text(0.105, 0.965, "Harvesting strategies", fontsize=23,
             fontweight="bold", color="#243847")
    fig.text(0.105, 0.934,
             rf"$m={m:g}\quad K_A={k_a:g}\quad K_B={k_b:g}$"
             "     Simulation-based regions",
             fontsize=13.5, color="#596975")
    inset_key = np.all(zones[equation_card_mask(r_a_values, r_b_values)] == INTERIOR)
    add_map_labels(ax, r_a_values, r_b_values, zones, markers, inset_key=inset_key)
    if inset_key:
        add_equation_key(ax, k_a=k_a, k_b=k_b)
    else:
        # Keep the equations outside when other parameters put a boundary here.
        fig.set_size_inches(13.5, 9.0)
        ax.set_position((0.065, 0.08, 0.56, 0.83))
        key_ax = fig.add_axes((0.64, 0.24, 0.35, 0.66))
        key_ax.set(xlim=(0.27, 0.995), ylim=(0.395, 0.995))
        key_ax.axis("off")
        add_equation_key(key_ax, k_a=k_a, k_b=k_b, transform=key_ax.transData)

    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
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
    parser.add_argument("--local-effort-grid-size", type=int, default=9)
    parser.add_argument("--effort-tolerance", type=float, default=1e-5)
    parser.add_argument("--yield-atol", type=float, default=1e-12)
    parser.add_argument("--yield-rtol", type=float, default=1e-9)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--yield-tolerance", type=float, default=1e-10)
    parser.add_argument("--classification-tolerance", type=float, default=1e-4)
    parser.add_argument("--data-input", type=Path,
                        help="Replot a saved NPZ grid without rerunning the numerical search.")
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
    if args.data_input:
        with np.load(args.data_input, allow_pickle=False) as data:
            if "algorithm" not in data or str(data["algorithm"]) != ALGORITHM:
                raise ValueError("Saved data uses an older solver; regenerate without --data-input.")
            for name in ("m", "k_a", "k_b"):
                if not np.isclose(float(data[name]), getattr(args, name)):
                    raise ValueError(f"Saved {name} does not match the requested parameter.")
            r_a_values, r_b_values, zones = [data[name].copy() for name in
                                            ("r_a_values", "r_b_values", "zones")]
        if zones.shape != (len(r_b_values), len(r_a_values)) or not np.isin(zones, np.arange(8)).all():
            raise ValueError("Invalid saved classification grid.")
    else:
        r_a_values, r_b_values, zones = create_numeric_zone_grid(
            grid_size=args.grid_size,
            r_min=args.r_min,
            r_max=args.r_max,
            m=args.m,
            k_a=args.k_a,
            k_b=args.k_b,
            effort_grid_size=args.effort_grid_size,
            local_effort_grid_size=args.local_effort_grid_size,
            effort_tolerance=args.effort_tolerance,
            yield_atol=args.yield_atol,
            yield_rtol=args.yield_rtol,
            max_iterations=args.max_iterations,
            processes=args.processes,
            yield_tolerance=args.yield_tolerance,
            classification_tolerance=args.classification_tolerance,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_suffix(".npz"), r_a_values=r_a_values,
                            r_b_values=r_b_values, zones=zones, m=args.m,
                            k_a=args.k_a, k_b=args.k_b,
                            algorithm=ALGORITHM,
                            effort_grid_size=args.effort_grid_size,
                            local_effort_grid_size=args.local_effort_grid_size,
                            effort_tolerance=args.effort_tolerance,
                            yield_atol=args.yield_atol, yield_rtol=args.yield_rtol,
                            max_iterations=args.max_iterations,
                            yield_tolerance=args.yield_tolerance,
                            classification_tolerance=args.classification_tolerance)
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
    print(f"grid size: {len(r_a_values)} x {len(r_b_values)}")
    if args.data_input:
        print(f"reused numerical grid: {args.data_input}")
    else:
        print(f"coarse effort grid: {args.effort_grid_size} x {args.effort_grid_size}")
        print(f"local effort grid: {args.local_effort_grid_size} x {args.local_effort_grid_size}")
        print(f"effort resolution: {args.effort_tolerance:g}")
        print(f"yield bounds: atol={args.yield_atol:g}, rtol={args.yield_rtol:g}")
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
