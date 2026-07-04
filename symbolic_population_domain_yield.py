from __future__ import annotations

import sympy as sp


N_A, N_B = sp.symbols("N_A N_B", nonnegative=True)
K_A, K_B = sp.symbols("K_A K_B", positive=True)
r_A, r_B = sp.symbols("r_A r_B", positive=True)
m = sp.symbols("m", nonnegative=True)
q_A, q_B = sp.symbols("q_A q_B", positive=True)


def beverton_holt(population: sp.Expr, r: sp.Expr, k: sp.Expr) -> sp.Expr:
    return k * r * population / (k + population)


F_A = beverton_holt(N_A, r_A, K_A)
F_B = beverton_holt(N_B, r_B, K_B)
G_A = (1 - m) * F_A + m * F_B
G_B = m * F_A + (1 - m) * F_B

population_domain_yield = sp.factor(F_A + F_B - N_A - N_B)
constraint_a = sp.factor(N_A - G_A)
constraint_b = sp.factor(N_B - G_B)


def interior_solution() -> dict[str, sp.Expr]:
    derivative_a = sp.diff(F_A - N_A, N_A)
    derivative_b = sp.diff(F_B - N_B, N_B)
    solution_a = sp.solve(sp.Eq(derivative_a, 0), N_A)[0]
    solution_b = sp.solve(sp.Eq(derivative_b, 0), N_B)[0]
    return {
        "d/dN_A(F_A-N_A)": sp.factor(derivative_a),
        "d/dN_B(F_B-N_B)": sp.factor(derivative_b),
        "N_A*": sp.factor(solution_a),
        "N_B*": sp.factor(solution_b),
    }


def concavity_and_constraints() -> dict[str, sp.Expr]:
    return {
        "F_A''": sp.factor(sp.diff(F_A, N_A, 2)),
        "F_B''": sp.factor(sp.diff(F_B, N_B, 2)),
        "Hessian(Y)": sp.hessian(population_domain_yield, (N_A, N_B)),
        "constraint A": constraint_a,
        "constraint B": constraint_b,
        "constraint A Hessian": sp.hessian(constraint_a, (N_A, N_B)),
        "constraint B Hessian": sp.hessian(constraint_b, (N_A, N_B)),
    }


def complete_harvest_edge() -> dict[str, sp.Expr]:
    n = sp.symbols("n", nonnegative=True)
    growth = beverton_holt(n, r_B, K_B)
    edge_yield = sp.factor(growth - n)
    edge_capacity = sp.factor(K_B * ((1 - m) * r_B - 1))
    unconstrained = sp.factor(K_B * (sp.sqrt(r_B) - 1))
    effort_b = sp.factor(1 - n / ((1 - m) * growth))
    return {
        "edge yield with N_A = 0": edge_yield,
        "edge upper bound for N_B": edge_capacity,
        "unconstrained edge maximizer": unconstrained,
        "effort e_B from N_B": effort_b,
        "partial-harvest feasibility": (1 - m) * sp.sqrt(r_B) - 1,
    }


def protected_source_endpoint_multiplier() -> dict[str, sp.Expr]:
    q_source = 1 / ((1 - m) ** 2 * r_B)
    lambda_source = sp.factor(
        (q_source - 1) / (1 - (1 - m) * q_source)
    )
    sink_condition = sp.factor(1 / (1 + m * lambda_source))
    return {
        "q_B at protected-source endpoint": q_source,
        "lambda_B": lambda_source,
        "condition on r_A": sink_condition,
    }


def no_take_stationarity() -> dict[str, sp.Expr]:
    lambda_a = sp.symbols("lambda_a", nonnegative=True)
    equation_a = sp.factor(1 - q_A + lambda_a * (1 - (1 - m) * q_A))
    equation_b = sp.factor(1 - q_B - lambda_a * m * q_B)
    lambda_from_b = sp.solve(sp.Eq(equation_b, 0), lambda_a)[0]
    reduced = sp.factor(equation_a.subs(lambda_a, lambda_from_b))
    stationarity = sp.factor(sp.together(reduced).as_numer_denom()[0])
    expected = sp.factor((1 - m) * (q_A + q_B) + (2 * m - 1) * q_A * q_B - 1)
    return {
        "KKT equation A": equation_a,
        "KKT equation B": equation_b,
        "lambda_A from equation B": sp.factor(lambda_from_b),
        "stationarity": expected,
        "check up to sign": sp.factor(stationarity + expected),
    }


def print_section(title: str, expressions: dict[str, sp.Expr]) -> None:
    print(title)
    print("=" * len(title))
    for name, expression in expressions.items():
        print(f"{name}:")
        print(sp.latex(expression))
        print()


def main() -> None:
    print_section("Population-domain yield", {"Y": population_domain_yield})
    print_section("Concavity and feasible constraints", concavity_and_constraints())
    print_section("Interior solution", interior_solution())
    print_section("Complete-harvest edge N_A=0", complete_harvest_edge())
    print_section(
        "Protected-source endpoint N_A=0, e_B=0",
        protected_source_endpoint_multiplier(),
    )
    print_section("No-take stationarity", no_take_stationarity())


if __name__ == "__main__":
    main()
