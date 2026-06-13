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


def beverton_holt_map() -> sp.Expr:
    return K * r * N / (K + N)


def beverton_holt_map_with_fishing() -> sp.Expr:
    return (1 - e) * beverton_holt_map()


def analyze_equilibria(
    name: str,
    map_expression: sp.Expr,
    positive_equilibrium: sp.Expr,
) -> EquilibriumAnalysis:
    derivative = sp.diff(map_expression, N)

    return EquilibriumAnalysis(
        name=name,
        map_expression=map_expression,
        equilibria=sp.solve(sp.factor(map_expression - N), N),
        derivative=sp.simplify(derivative),
        extinction_derivative=sp.simplify(derivative.subs(N, 0)),
        positive_equilibrium=positive_equilibrium,
        positive_equilibrium_derivative=sp.simplify(
            derivative.subs(N, positive_equilibrium)
        ),
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


def main() -> None:
    base_analysis = analyze_equilibria(
        name="Beverton-Holt map",
        map_expression=beverton_holt_map(),
        positive_equilibrium=K * (r - 1),
    )
    fishing_analysis = analyze_equilibria(
        name="Beverton-Holt map with fishing effort",
        map_expression=beverton_holt_map_with_fishing(),
        positive_equilibrium=K * (r * (1 - e) - 1),
    )

    show_equilibrium_analysis(base_analysis)
    show_equilibrium_analysis(fishing_analysis)


if __name__ == "__main__":
    main()
