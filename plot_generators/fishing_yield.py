from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_generators.common import (
    K,
    PLOTS_DIR,
    R,
    equilibrium_yield,
    extinction_effort,
    optimal_yield_effort,
    save_figure,
    simulate_effort_range,
)


OUTPUT_PATH = PLOTS_DIR / "fishing_yield_r1_4_k1.pdf"


def create_plot(r: float, k: float, output_path: Path) -> None:
    critical_effort = extinction_effort(r)
    optimal_effort = optimal_yield_effort(r)
    efforts = np.linspace(0, 1, 500)
    yields = equilibrium_yield(efforts, r, k)
    simulation_efforts, _, simulation_yields = simulate_effort_range(r, k)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(efforts, yields, linewidth=2, label=r"$Y(e)$")
    ax.scatter(
        simulation_efforts,
        simulation_yields,
        color="black",
        s=12,
        alpha=0.7,
        label="simulation",
    )
    ax.axvline(
        optimal_effort,
        color="green",
        linestyle="--",
        linewidth=2,
        label=rf"$e^*=1-\frac{{1}}{{\sqrt{{r}}}}\approx {optimal_effort:.3f}$",
    )
    ax.axvline(
        critical_effort,
        color="red",
        linestyle="--",
        linewidth=2,
        label=rf"$e_c=1-\frac{{1}}{{r}}\approx {critical_effort:.3f}$",
    )
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Fishing effort, $e$")
    ax.set_ylabel("Equilibrium yield, $Y(e)$")
    ax.set_title(rf"Fishing yield for $r={r}$, $K={k}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    create_plot(r=R, k=K, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
