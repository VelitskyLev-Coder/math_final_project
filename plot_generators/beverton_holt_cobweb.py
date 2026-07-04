from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from plot_generators.common import K, PLOTS_DIR, R, beverton_holt_growth, save_figure


OUTPUT_PATH = PLOTS_DIR / "beverton_holt_cobweb_r1_4_k1.pdf"


def cobweb_points(
    initial_population: float,
    r: float,
    k: float,
    steps: int,
) -> tuple[list[float], list[float]]:
    populations = [initial_population]
    for _ in range(steps):
        populations.append(beverton_holt_growth(populations[-1], r, k))

    x_values = [populations[0]]
    y_values = [0.0]
    for previous_population, next_population in zip(populations[:-1], populations[1:]):
        x_values.extend([previous_population, next_population])
        y_values.extend([next_population, next_population])

    return x_values, y_values


def create_plot(r: float, k: float, output_path: Path) -> None:
    initial_population = 0.1
    steps = 22
    positive_equilibrium = k * (r - 1)
    max_population = 0.75
    populations = np.linspace(0, max_population, 500)
    growth = beverton_holt_growth(populations, r, k)
    cobweb_x, cobweb_y = cobweb_points(initial_population, r, k, steps)
    zoom_min = 0.37
    zoom_max = 0.41

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(populations, growth, linewidth=2, label=r"$F(N)$")
    ax.plot(populations, populations, color="black", linewidth=1.2, label=r"$N_{t+1}=N_t$")
    ax.plot(cobweb_x, cobweb_y, color="tab:red", linewidth=1.5, alpha=0.9, label="iteration")
    ax.scatter(
        [positive_equilibrium],
        [positive_equilibrium],
        color="black",
        s=35,
        zorder=3,
        label=rf"$N^*=K(r-1)={positive_equilibrium:.2f}$",
    )

    ax.set_xlabel(r"$N_t$")
    ax.set_ylabel(r"$N_{t+1}$")
    ax.set_title(rf"Cobweb plot for $r={r}$, $K={k}$")
    ax.set_xlim(0, max_population)
    ax.set_ylim(0, max_population)
    ax.grid(True, alpha=0.3)
    ax.legend()

    inset_ax = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=1.2)
    inset_ax.plot(populations, growth, linewidth=2)
    inset_ax.plot(populations, populations, color="black", linewidth=1.2)
    inset_ax.plot(cobweb_x, cobweb_y, color="tab:red", linewidth=1.5, alpha=0.9)
    inset_ax.scatter([positive_equilibrium], [positive_equilibrium], color="black", s=25, zorder=3)
    inset_ax.set_xlim(zoom_min, zoom_max)
    inset_ax.set_ylim(zoom_min, zoom_max)
    inset_ax.set_xticks([zoom_min, positive_equilibrium, zoom_max])
    inset_ax.set_yticks([zoom_min, positive_equilibrium, zoom_max])
    inset_ax.tick_params(labelsize=7)
    inset_ax.grid(True, alpha=0.25)
    mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.35", linewidth=1.0)

    save_figure(fig, output_path, tight_layout=False)
    plt.close(fig)


def main() -> None:
    create_plot(r=R, k=K, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
