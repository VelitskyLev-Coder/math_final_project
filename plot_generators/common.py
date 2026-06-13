from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np


SIMULATION_POINTS = 100
SIMULATION_STEPS = 1000
INITIAL_POPULATION = 0.5
R = 1.4
K = 1
PLOTS_DIR = Path("tex/plots")
LINKED_K_A = 1
LINKED_K_B = 1
LINKED_INITIAL_A = 0.5
LINKED_INITIAL_B = 0.5
LINKED_SIMULATION_STEPS = 1000
LINKED_EXTINCTION_EPSILON = 1e-4


def beverton_holt_growth(population: float, r: float, k: float) -> float:
    return k * r * population / (k + population)


def simulate_equilibrium(
    effort: float,
    r: float,
    k: float,
    steps: int = SIMULATION_STEPS,
    initial_population: float = INITIAL_POPULATION,
) -> tuple[float, float]:
    population = initial_population
    yield_value = 0.0

    for _ in range(steps):
        grown_population = beverton_holt_growth(population, r, k)
        yield_value = effort * grown_population
        population = (1 - effort) * grown_population

    return population, yield_value


def simulate_effort_range(r: float, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    efforts = np.linspace(0, 1, SIMULATION_POINTS)
    populations = np.zeros_like(efforts)
    yields = np.zeros_like(efforts)

    for index, effort in enumerate(efforts):
        populations[index], yields[index] = simulate_equilibrium(effort, r, k)

    return efforts, populations, yields


def equilibrium_with_fishing(effort: np.ndarray, r: float, k: float) -> np.ndarray:
    positive_equilibrium = k * (r * (1 - effort) - 1)
    return np.maximum(positive_equilibrium, 0)


def extinction_effort(r: float) -> float:
    if r <= 1:
        return 0

    return 1 - 1 / r


def equilibrium_yield(effort: np.ndarray, r: float, k: float) -> np.ndarray:
    yield_values = np.zeros_like(effort)
    valid_effort = effort < extinction_effort(r)
    yield_values[valid_effort] = (
        k
        * effort[valid_effort]
        * (r - 1 / (1 - effort[valid_effort]))
    )
    return yield_values


def optimal_yield_effort(r: float) -> float:
    if r <= 1:
        return 0

    return 1 - 1 / np.sqrt(r)


def save_figure(fig, output_path: Path, tight_layout: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    fig.savefig(output_path)


def linked_beverton_holt_growth(population: float, r: float, k: float) -> float:
    return k * r * population / (k + population)


def linked_next_population(
    population_a: float,
    population_b: float,
    r_a: float,
    r_b: float,
    m: float,
    e_a: float,
    e_b: float,
    k_a: float = LINKED_K_A,
    k_b: float = LINKED_K_B,
) -> tuple[float, float]:
    growth_a = linked_beverton_holt_growth(population_a, r_a, k_a)
    growth_b = linked_beverton_holt_growth(population_b, r_b, k_b)

    next_a = (1 - e_a) * ((1 - m) * growth_a + m * growth_b)
    next_b = (1 - e_b) * (m * growth_a + (1 - m) * growth_b)
    return next_a, next_b


def simulate_linked_total_population(
    r_a: float,
    r_b: float,
    m: float,
    e_a: float,
    e_b: float,
    steps: int = LINKED_SIMULATION_STEPS,
) -> float:
    population_a = LINKED_INITIAL_A
    population_b = LINKED_INITIAL_B

    for _ in range(steps):
        population_a, population_b = linked_next_population(
            population_a=population_a,
            population_b=population_b,
            r_a=r_a,
            r_b=r_b,
            m=m,
            e_a=e_a,
            e_b=e_b,
        )

    return population_a + population_b


def simulate_linked_equilibrium_and_yield(
    r_a: float,
    r_b: float,
    m: float,
    e_a: float,
    e_b: float,
    k_a: float = LINKED_K_A,
    k_b: float = LINKED_K_B,
    steps: int = LINKED_SIMULATION_STEPS,
) -> tuple[float, float, float]:
    population_a = LINKED_INITIAL_A
    population_b = LINKED_INITIAL_B
    yield_value = 0.0

    for _ in range(steps):
        growth_a = linked_beverton_holt_growth(population_a, r_a, k_a)
        growth_b = linked_beverton_holt_growth(population_b, r_b, k_b)
        pre_fishing_a = (1 - m) * growth_a + m * growth_b
        pre_fishing_b = m * growth_a + (1 - m) * growth_b
        yield_value = e_a * pre_fishing_a + e_b * pre_fishing_b
        population_a = (1 - e_a) * pre_fishing_a
        population_b = (1 - e_b) * pre_fishing_b

    return population_a, population_b, yield_value


def linked_fishing_spectral_radius(
    e_a: np.ndarray,
    e_b: np.ndarray,
    r_a: float,
    r_b: float,
    m: float,
) -> np.ndarray:
    matrix_a = r_a * (1 - m) * (1 - e_a)
    matrix_b = r_b * m * (1 - e_a)
    matrix_c = r_a * m * (1 - e_b)
    matrix_d = r_b * (1 - m) * (1 - e_b)

    trace = matrix_a + matrix_d
    discriminant = (matrix_a - matrix_d) ** 2 + 4 * matrix_b * matrix_c
    return (trace + np.sqrt(discriminant)) / 2


def linked_spectral_radius(
    r_a: np.ndarray,
    r_b: np.ndarray,
    m: float,
) -> np.ndarray:
    matrix_a = r_a * (1 - m)
    matrix_b = r_b * m
    matrix_c = r_a * m
    matrix_d = r_b * (1 - m)

    trace = matrix_a + matrix_d
    discriminant = (matrix_a - matrix_d) ** 2 + 4 * matrix_b * matrix_c
    return (trace + np.sqrt(discriminant)) / 2
