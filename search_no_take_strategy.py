import argparse
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar


@dataclass(frozen=True)
class SearchConfig:
    samples: int
    grid_size: int
    simulation_steps: int
    seed: int
    r_min: float
    r_max: float
    m_min: float
    m_max: float
    refine_limit: int


@dataclass(frozen=True)
class Candidate:
    r_a: float
    r_b: float
    m: float
    grid_e_a: float
    grid_e_b: float
    grid_yield: float
    edge: str


def beverton_holt(population: np.ndarray, r: float, k: float = 1.0) -> np.ndarray:
    return k * r * population / (k + population)


def yield_grid(
    r_a: float,
    r_b: float,
    m: float,
    grid_size: int,
    simulation_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    efforts = np.linspace(0.0, 1.0, grid_size)
    effort_a, effort_b = np.meshgrid(efforts, efforts)
    population_a = np.full_like(effort_a, 0.5, dtype=float)
    population_b = np.full_like(effort_b, 0.5, dtype=float)
    yield_values = np.zeros_like(effort_a, dtype=float)

    for _ in range(simulation_steps):
        growth_a = beverton_holt(population_a, r_a)
        growth_b = beverton_holt(population_b, r_b)
        pre_fishing_a = (1 - m) * growth_a + m * growth_b
        pre_fishing_b = m * growth_a + (1 - m) * growth_b
        yield_values = effort_a * pre_fishing_a + effort_b * pre_fishing_b
        population_a = (1 - effort_a) * pre_fishing_a
        population_b = (1 - effort_b) * pre_fishing_b

    return effort_a, effort_b, yield_values


def simulate_yield(
    r_a: float,
    r_b: float,
    m: float,
    e_a: float,
    e_b: float,
    simulation_steps: int,
) -> float:
    population_a = 0.5
    population_b = 0.5
    yield_value = 0.0

    for _ in range(simulation_steps):
        growth_a = float(beverton_holt(np.array(population_a), r_a))
        growth_b = float(beverton_holt(np.array(population_b), r_b))
        pre_fishing_a = (1 - m) * growth_a + m * growth_b
        pre_fishing_b = m * growth_a + (1 - m) * growth_b
        yield_value = e_a * pre_fishing_a + e_b * pre_fishing_b
        population_a = (1 - e_a) * pre_fishing_a
        population_b = (1 - e_b) * pre_fishing_b

    return yield_value


def classify_grid_winner(
    r_a: float,
    r_b: float,
    m: float,
    config: SearchConfig,
) -> Candidate | None:
    effort_a, effort_b, yields = yield_grid(
        r_a,
        r_b,
        m,
        grid_size=config.grid_size,
        simulation_steps=config.simulation_steps,
    )
    row, column = np.unravel_index(np.nanargmax(yields), yields.shape)
    e_a = float(effort_a[row, column])
    e_b = float(effort_b[row, column])
    yield_value = float(yields[row, column])

    if column == 0 and 0 < row < config.grid_size - 1:
        return Candidate(r_a, r_b, m, e_a, e_b, yield_value, "e_A = 0")
    if row == 0 and 0 < column < config.grid_size - 1:
        return Candidate(r_a, r_b, m, e_a, e_b, yield_value, "e_B = 0")

    return None


def refine_candidate(candidate: Candidate, simulation_steps: int) -> dict[str, float]:
    def negative_full(point: np.ndarray) -> float:
        return -simulate_yield(
            candidate.r_a,
            candidate.r_b,
            candidate.m,
            float(point[0]),
            float(point[1]),
            simulation_steps,
        )

    def negative_edge_a(e_b: float) -> float:
        return -simulate_yield(
            candidate.r_a,
            candidate.r_b,
            candidate.m,
            0.0,
            e_b,
            simulation_steps,
        )

    def negative_edge_b(e_a: float) -> float:
        return -simulate_yield(
            candidate.r_a,
            candidate.r_b,
            candidate.m,
            e_a,
            0.0,
            simulation_steps,
        )

    full_result = differential_evolution(
        negative_full,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        seed=123,
        polish=True,
        tol=1e-7,
        maxiter=80,
        popsize=10,
    )
    edge_a_result = minimize_scalar(negative_edge_a, bounds=(0.0, 1.0), method="bounded")
    edge_b_result = minimize_scalar(negative_edge_b, bounds=(0.0, 1.0), method="bounded")

    return {
        "full_e_a": float(full_result.x[0]),
        "full_e_b": float(full_result.x[1]),
        "full_yield": float(-full_result.fun),
        "edge_a_e_b": float(edge_a_result.x),
        "edge_a_yield": float(-edge_a_result.fun),
        "edge_b_e_a": float(edge_b_result.x),
        "edge_b_yield": float(-edge_b_result.fun),
    }


def search(config: SearchConfig) -> list[Candidate]:
    rng = np.random.default_rng(config.seed)
    candidates: list[Candidate] = []

    for _ in range(config.samples):
        r_a = float(rng.uniform(config.r_min, config.r_max))
        r_b = float(rng.uniform(config.r_min, config.r_max))
        m = float(rng.uniform(config.m_min, config.m_max))
        candidate = classify_grid_winner(r_a, r_b, m, config)
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--simulation-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--r-min", type=float, default=0.2)
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--m-min", type=float, default=0.01)
    parser.add_argument("--m-max", type=float, default=0.99)
    parser.add_argument("--refine-limit", type=int, default=10)
    args = parser.parse_args()

    config = SearchConfig(**vars(args))
    candidates = search(config)

    print("No-take strategy search")
    print("=======================")
    print(f"samples: {config.samples}")
    print(f"grid: {config.grid_size} x {config.grid_size}")
    print(f"simulation steps: {config.simulation_steps}")
    print(f"r range: [{config.r_min}, {config.r_max}]")
    print(f"m range: [{config.m_min}, {config.m_max}]")
    print(f"strict no-take grid winners: {len(candidates)}")
    if candidates:
        edge_counts = Counter(candidate.edge for candidate in candidates)
        both_persistent = sum(
            1 for candidate in candidates if candidate.r_a > 1 and candidate.r_b > 1
        )
        print(f"  e_A = 0 winners: {edge_counts['e_A = 0']}")
        print(f"  e_B = 0 winners: {edge_counts['e_B = 0']}")
        print(f"  with r_A > 1 and r_B > 1: {both_persistent}")

    if not candidates:
        return

    print()
    print("Refined examples")
    print("================")
    for index, candidate in enumerate(candidates[: config.refine_limit], start=1):
        refined = refine_candidate(candidate, config.simulation_steps)
        print(
            f"{index}. {candidate.edge}: "
            f"r_A={candidate.r_a:.4f}, r_B={candidate.r_b:.4f}, m={candidate.m:.4f}"
        )
        print(
            f"   grid best: e_A={candidate.grid_e_a:.4f}, "
            f"e_B={candidate.grid_e_b:.4f}, Y={candidate.grid_yield:.6f}"
        )
        print(
            f"   full optimum: e_A={refined['full_e_a']:.6f}, "
            f"e_B={refined['full_e_b']:.6f}, Y={refined['full_yield']:.6f}"
        )
        print(
            f"   e_A=0 edge: e_B={refined['edge_a_e_b']:.6f}, "
            f"Y={refined['edge_a_yield']:.6f}"
        )
        print(
            f"   e_B=0 edge: e_A={refined['edge_b_e_a']:.6f}, "
            f"Y={refined['edge_b_yield']:.6f}"
        )


if __name__ == "__main__":
    main()
