from dataclasses import dataclass


@dataclass(frozen=True)
class BevertonHoltModel:
    """Beverton-Holt population map with an explicit population scale."""

    r: float
    k: float

    def __post_init__(self) -> None:
        if self.r < 0:
            raise ValueError("r must be non-negative")
        if self.k <= 0:
            raise ValueError("k must be positive")

    def next_population(self, population: float) -> float:
        if population < 0:
            raise ValueError("population must be non-negative")

        return self.k * self.r * population / (self.k + population)

    @property
    def positive_equilibrium(self) -> float | None:
        """Return the positive equilibrium, which exists only when r > 1."""
        if self.r <= 1:
            return None

        return self.k * (self.r - 1)
