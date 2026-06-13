from dataclasses import dataclass

import sympy as sp


N, K, r, e = sp.symbols("N K r e", nonnegative=True)


@dataclass(frozen=True)
class EquilibriumAnalysis:
    name: str
    map_expression: sp.Expr
    equilibria: list[sp.Expr]
    derivative: sp.Expr
    extinction_derivative: sp.Expr
    positive_equilibrium: sp.Expr
    positive_equilibrium_derivative: sp.Expr


@dataclass(frozen=True)
class YieldAnalysis:
    name: str
    positive_equilibrium: sp.Expr
    yield_expression: sp.Expr
    yield_derivative: sp.Expr
    critical_efforts: list[sp.Expr]
    admissible_maximizer: sp.Expr


def beverton_holt_map() -> sp.Expr:
    return K * r * N / (K + N)


def beverton_holt_map_with_fishing() -> sp.Expr:
    return (1 - e) * beverton_holt_map()


def analyze_equilibria(
    name: str,
    map_expression: sp.Expr,
) -> EquilibriumAnalysis:
    derivative = sp.diff(map_expression, N)
    equilibria = sp.solve(sp.factor(map_expression - N), N)
    positive_equilibrium = nonzero_equilibrium(equilibria)

    return EquilibriumAnalysis(
        name=name,
        map_expression=map_expression,
        equilibria=equilibria,
        derivative=sp.simplify(derivative),
        extinction_derivative=sp.simplify(derivative.subs(N, 0)),
        positive_equilibrium=positive_equilibrium,
        positive_equilibrium_derivative=sp.simplify(
            derivative.subs(N, positive_equilibrium)
        ),
    )


def nonzero_equilibrium(equilibria: list[sp.Expr]) -> sp.Expr:
    nonzero_equilibria = [
        equilibrium
        for equilibrium in equilibria
        if sp.simplify(equilibrium) != 0
    ]
    if len(nonzero_equilibria) != 1:
        raise ValueError(f"Expected one nonzero equilibrium, got {nonzero_equilibria}")

    return sp.simplify(nonzero_equilibria[0])


def equilibrium_yield_expression(positive_equilibrium: sp.Expr) -> sp.Expr:
    return sp.simplify(e * positive_equilibrium / (1 - e))


def analyze_yield(
    name: str,
    equilibrium_analysis: EquilibriumAnalysis,
) -> YieldAnalysis:
    positive_equilibrium = equilibrium_analysis.positive_equilibrium
    yield_expression = equilibrium_yield_expression(positive_equilibrium)
    yield_derivative = sp.diff(yield_expression, e)

    return YieldAnalysis(
        name=name,
        positive_equilibrium=positive_equilibrium,
        yield_expression=sp.simplify(yield_expression),
        yield_derivative=sp.simplify(yield_derivative),
        critical_efforts=sp.solve(sp.factor(yield_derivative), e),
        admissible_maximizer=1 - 1 / sp.sqrt(r),
    )


def as_latex(expression: sp.Expr) -> str:
    return sp.latex(sp.simplify(expression))


def readable_expression(expression: sp.Expr) -> sp.Expr:
    return sp.factor_terms(sp.simplify(expression), sign=False)


def readable_latex(expression: sp.Expr) -> str:
    return sp.latex(readable_expression(expression))


def show_equilibrium_analysis(analysis: EquilibriumAnalysis) -> None:
    print(analysis.name)
    print("=" * len(analysis.name))
    print(f"F(N) = {readable_latex(analysis.map_expression)}")
    print()

    print("Equilibrium equation:")
    print(
        "F(N) = N  <=>  "
        f"{readable_latex(sp.factor(analysis.map_expression - N))} = 0"
    )
    print()

    print("Equilibria:")
    for equilibrium in analysis.equilibria:
        print(f"N* = {readable_latex(equilibrium)}")
    print()

    print("Derivative:")
    print(f"F'(N) = {readable_latex(analysis.derivative)}")
    print()

    print("Derivative at extinction:")
    print(f"F'(0) = {readable_latex(analysis.extinction_derivative)}")
    print()

    print("Derivative at positive equilibrium:")
    print(
        f"F'({readable_latex(analysis.positive_equilibrium)}) = "
        f"{readable_latex(analysis.positive_equilibrium_derivative)}"
    )
    print()


def show_yield_analysis(analysis: YieldAnalysis) -> None:
    print(analysis.name)
    print("=" * len(analysis.name))
    print(f"N* = {readable_latex(analysis.positive_equilibrium)}")
    print()

    print("Equilibrium yield:")
    print(f"Y(e) = eN*/(1-e) = {readable_latex(analysis.yield_expression)}")
    print()

    print("Yield derivative:")
    print(f"Y'(e) = {readable_latex(analysis.yield_derivative)}")
    print()

    print("Critical efforts:")
    for critical_effort in analysis.critical_efforts:
        print(f"e* = {readable_latex(critical_effort)}")
    print()

    print("Admissible maximizer for 0 <= e <= 1:")
    print(f"e* = {readable_latex(analysis.admissible_maximizer)}")
    print()


def main() -> None:
    base_analysis = analyze_equilibria(
        name="Beverton-Holt map",
        map_expression=beverton_holt_map(),
    )
    fishing_analysis = analyze_equilibria(
        name="Beverton-Holt map with fishing effort",
        map_expression=beverton_holt_map_with_fishing(),
    )
    yield_analysis = analyze_yield(
        name="Maximizing fishing yield",
        equilibrium_analysis=fishing_analysis,
    )

    #show_equilibrium_analysis(base_analysis)
    show_equilibrium_analysis(fishing_analysis)
    show_yield_analysis(yield_analysis)


if __name__ == "__main__":
    main()
