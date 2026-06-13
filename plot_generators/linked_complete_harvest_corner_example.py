import matplotlib.pyplot as plt
import numpy as np

from plot_generators.common import (
    PLOTS_DIR,
    save_figure,
    simulate_linked_equilibrium_and_yield,
)


GRID_POINTS = 101
SIMULATION_STEPS = 800
OUTPUT_PATH = PLOTS_DIR / "linked_complete_harvest_corner_example.pdf"

EXAMPLES = (
    {
        "title": r"Protect $A$, harvest $B$",
        "r_a": 4.0,
        "r_b": 0.8,
        "m": 0.5,
        "candidate_e_a": 0.0,
        "candidate_e_b": 1.0,
        "edge_label": r"$e_B=1$",
        "edge": "B",
    },
    {
        "title": r"Protect $B$, harvest $A$",
        "r_a": 0.8,
        "r_b": 1 / (1 - 0.3) ** 2,
        "m": 0.3,
        "candidate_e_a": 1.0,
        "candidate_e_b": 0.0,
        "edge_label": r"$e_A=1$",
        "edge": "A",
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


def create_subplot(ax, example: dict[str, float | str]) -> None:
    r_a = float(example["r_a"])
    r_b = float(example["r_b"])
    m = float(example["m"])
    candidate_e_a = float(example["candidate_e_a"])
    candidate_e_b = float(example["candidate_e_b"])
    effort_a_grid, effort_b_grid, yields = yield_grid(r_a, r_b, m)
    _, _, candidate_yield = simulate_linked_equilibrium_and_yield(
        r_a=r_a,
        r_b=r_b,
        m=m,
        e_a=candidate_e_a,
        e_b=candidate_e_b,
        steps=SIMULATION_STEPS,
    )
    max_index = np.unravel_index(np.nanargmax(yields), yields.shape)
    grid_e_a = float(effort_a_grid[max_index])
    grid_e_b = float(effort_b_grid[max_index])

    heatmap = ax.contourf(
        effort_a_grid,
        effort_b_grid,
        yields,
        levels=np.linspace(0, np.nanmax(yields), 31),
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

    if str(example["edge"]) == "A":
        ax.axvline(1, color="red", linewidth=2.2, alpha=0.9, label=example["edge_label"])
    else:
        ax.axhline(1, color="red", linewidth=2.2, alpha=0.9, label=example["edge_label"])

    ax.scatter(
        [candidate_e_a],
        [candidate_e_b],
        color="red",
        edgecolor="white",
        linewidth=1.2,
        s=80,
        zorder=4,
        clip_on=False,
        label="analytic endpoint candidate",
    )
    ax.scatter(
        [grid_e_a],
        [grid_e_b],
        marker="x",
        color="black",
        s=80,
        linewidth=2,
        zorder=5,
        clip_on=False,
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
    ax.legend(loc="lower left", fontsize=8)

    return heatmap


def create_plot() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    heatmap = None
    for ax, example in zip(axes, EXAMPLES):
        heatmap = create_subplot(ax, example)

    if heatmap is not None:
        colorbar = fig.colorbar(heatmap, ax=axes, shrink=0.9)
        colorbar.set_label("equilibrium yield")

    save_figure(fig, OUTPUT_PATH, tight_layout=False)
    plt.close(fig)


def main() -> None:
    create_plot()


if __name__ == "__main__":
    main()
