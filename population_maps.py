import math
from typing import Protocol, NamedTuple


class GrowthModel(Protocol):
    def calculate_growth(self, N: float) -> float:
        pass


class BevertonHoltModel:
    def __init__(self, r):
        self.r = r

    def calculate_growth(self, N):
        return self.r * N / (1 + N)


class RickerModel:
    def __init__(self, r):
        self.r = r

    def calculate_growth(self, N):
        return self.r * N * math.exp(-N)


class TwoSubpopulations(NamedTuple):
    first_population_size: float
    second_population_size: float

    @property
    def total(self):
        return self.first_population_size + self.second_population_size


class TwoConnectedSubpopulationsModel:
    def __init__(self, growth_model_A: GrowthModel, growth_model_B: GrowthModel, m: float):
        self.growth_model_A = growth_model_A
        self.growth_model_B = growth_model_B
        self.m = m

    def calculate_next_population(self, populations: TwoSubpopulations) -> TwoSubpopulations:
        n_a = populations.first_population_size
        n_b = populations.second_population_size
        n_a_next = (1 - self.m) * self.growth_model_A.calculate_growth(
            n_a) + self.m * self.growth_model_B.calculate_growth(n_b)
        n_b_next = self.m * self.growth_model_A.calculate_growth(n_a) + (
                1 - self.m) * self.growth_model_B.calculate_growth(n_b)
        return TwoSubpopulations(n_a_next, n_b_next)
