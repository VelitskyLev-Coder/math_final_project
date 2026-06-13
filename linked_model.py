from dataclasses import dataclass

from beverton_holt import BevertonHoltModel


@dataclass(frozen=True)
class TwoPopulations:
    a: float
    b: float

    def __post_init__(self) -> None:
        if self.a < 0:
            raise ValueError("population a must be non-negative")
        if self.b < 0:
            raise ValueError("population b must be non-negative")

    @property
    def total(self) -> float:
        return self.a + self.b


@dataclass(frozen=True)
class LinkedBevertonHoltModel:
    population_a: BevertonHoltModel
    population_b: BevertonHoltModel
    migration_rate: float

    def __post_init__(self) -> None:
        if not 0 <= self.migration_rate <= 1:
            raise ValueError("migration_rate must be between 0 and 1")

    def next_population(self, populations: TwoPopulations) -> TwoPopulations:
        growth_a = self.population_a.next_population(populations.a)
        growth_b = self.population_b.next_population(populations.b)
        m = self.migration_rate

        return TwoPopulations(
            a=(1 - m) * growth_a + m * growth_b,
            b=m * growth_a + (1 - m) * growth_b,
        )

    def after_steps(self, populations: TwoPopulations, steps: int) -> TwoPopulations:
        if steps < 0:
            raise ValueError("steps must be non-negative")

        for _ in range(steps):
            populations = self.next_population(populations)

        return populations

    def equilibrium(
        self,
        initial_populations: TwoPopulations,
        tolerance: float = 1e-10,
        max_steps: int = 10000,
    ) -> TwoPopulations:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")

        populations = initial_populations
        for _ in range(max_steps):
            next_populations = self.next_population(populations)
            if (
                abs(next_populations.a - populations.a) <= tolerance
                and abs(next_populations.b - populations.b) <= tolerance
            ):
                return next_populations

            populations = next_populations

        raise RuntimeError("equilibrium iteration did not converge")
