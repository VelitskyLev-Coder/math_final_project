import sympy as sp


N_A, N_B = sp.symbols("N_A N_B", nonnegative=True)
K_A, K_B = sp.symbols("K_A K_B", positive=True)
r_A, r_B = sp.symbols("r_A r_B", nonnegative=True)
m = sp.symbols("m", nonnegative=True)
e_A, e_B = sp.symbols("e_A e_B", nonnegative=True)
lambda_A, lambda_B = sp.symbols("lambda_A lambda_B")


def beverton_holt_growth(population: sp.Symbol, growth_rate: sp.Symbol, scale: sp.Symbol) -> sp.Expr:
    return scale * growth_rate * population / (scale + population)


def linked_map() -> tuple[sp.Expr, sp.Expr]:
    growth_a = beverton_holt_growth(N_A, r_A, K_A)
    growth_b = beverton_holt_growth(N_B, r_B, K_B)

    next_a = (1 - m) * growth_a + m * growth_b
    next_b = m * growth_a + (1 - m) * growth_b
    return sp.simplify(next_a), sp.simplify(next_b)


def linked_map_with_fishing() -> tuple[sp.Expr, sp.Expr]:
    next_a, next_b = linked_map()
    return sp.simplify((1 - e_A) * next_a), sp.simplify((1 - e_B) * next_b)


def jacobian_at_extinction(map_expression: tuple[sp.Expr, sp.Expr]) -> sp.Matrix:
    jacobian = sp.Matrix(map_expression).jacobian([N_A, N_B])
    return sp.simplify(jacobian.subs({N_A: 0, N_B: 0}))


def extinction_boundary_expression(matrix: sp.Matrix) -> sp.Expr:
    identity = sp.eye(matrix.shape[0])
    return sp.factor((identity - matrix).det())


def trace_and_determinant(matrix: sp.Matrix) -> tuple[sp.Expr, sp.Expr]:
    return sp.factor(matrix.trace()), sp.factor(matrix.det())


def jury_stability_conditions(matrix: sp.Matrix) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    trace, determinant = trace_and_determinant(matrix)
    return (
        sp.factor(1 - trace + determinant),
        sp.factor(1 + trace + determinant),
        sp.factor(1 - determinant),
    )


def perron_eigenvalue(matrix: sp.Matrix) -> sp.Expr:
    trace, determinant = trace_and_determinant(matrix)
    discriminant = sp.factor(trace**2 - 4 * determinant)
    return sp.factor((trace + sp.sqrt(discriminant)) / 2)


def solve_extinction_boundary_for_e_b(matrix: sp.Matrix) -> sp.Expr:
    boundary = extinction_boundary_expression(matrix)
    return sp.solve(sp.factor(boundary), e_B)[0]


def solve_extinction_boundary_for_e_a(matrix: sp.Matrix) -> sp.Expr:
    boundary = extinction_boundary_expression(matrix)
    return sp.solve(sp.factor(boundary), e_A)[0]


def equilibrium_equations() -> tuple[sp.Expr, sp.Expr]:
    next_a, next_b = linked_map()
    return sp.factor(next_a - N_A), sp.factor(next_b - N_B)


def fishing_equilibrium_equations() -> tuple[sp.Expr, sp.Expr]:
    next_a, next_b = linked_map_with_fishing()
    return sp.factor(next_a - N_A), sp.factor(next_b - N_B)


def fishing_equilibrium_constraints() -> tuple[sp.Expr, sp.Expr]:
    next_a, next_b = linked_map_with_fishing()
    return sp.factor(N_A - next_a), sp.factor(N_B - next_b)


def linked_fishing_yield_expression() -> sp.Expr:
    return sp.factor(e_A * N_A / (1 - e_A) + e_B * N_B / (1 - e_B))


def linked_fishing_yield_lagrange_system() -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
    constraint_a, constraint_b = fishing_equilibrium_constraints()
    yield_expression = linked_fishing_yield_expression()
    lagrangian = yield_expression + lambda_A * constraint_a + lambda_B * constraint_b

    equations = (
        sp.factor(sp.diff(lagrangian, N_A)),
        sp.factor(sp.diff(lagrangian, N_B)),
        sp.factor(sp.diff(lagrangian, e_A)),
        sp.factor(sp.diff(lagrangian, e_B)),
        constraint_a,
        constraint_b,
    )
    return sp.factor(lagrangian), equations


def linked_fishing_yield_interior_candidate() -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    sqrt_r_a = sp.sqrt(r_A)
    sqrt_r_b = sp.sqrt(r_B)

    candidate_n_a = K_A * (sqrt_r_a - 1)
    candidate_n_b = K_B * (sqrt_r_b - 1)

    candidate_growth_a = beverton_holt_growth(candidate_n_a, r_A, K_A)
    candidate_growth_b = beverton_holt_growth(candidate_n_b, r_B, K_B)
    candidate_pre_fishing_a = (1 - m) * candidate_growth_a + m * candidate_growth_b
    candidate_pre_fishing_b = m * candidate_growth_a + (1 - m) * candidate_growth_b

    candidate_e_a = sp.factor(1 - candidate_n_a / candidate_pre_fishing_a)
    candidate_e_b = sp.factor(1 - candidate_n_b / candidate_pre_fishing_b)

    return (
        sp.factor(candidate_n_a),
        sp.factor(candidate_n_b),
        candidate_e_a,
        candidate_e_b,
    )


def linked_fishing_yield_reduced_interior_conditions() -> tuple[sp.Expr, sp.Expr]:
    constraint_a, constraint_b = fishing_equilibrium_constraints()
    yield_expression = linked_fishing_yield_expression()
    lagrangian = yield_expression + lambda_A * constraint_a + lambda_B * constraint_b

    multiplier_substitution = {
        lambda_A: -1 / (1 - e_A),
        lambda_B: -1 / (1 - e_B),
    }

    return (
        sp.factor(sp.diff(lagrangian, N_A).subs(multiplier_substitution)),
        sp.factor(sp.diff(lagrangian, N_B).subs(multiplier_substitution)),
    )


def linked_fishing_yield_boundary_conditions() -> dict[str, tuple[sp.Expr, ...]]:
    growth_a = beverton_holt_growth(N_A, r_A, K_A)
    growth_b = beverton_holt_growth(N_B, r_B, K_B)
    pre_fishing_a = (1 - m) * growth_a + m * growth_b
    pre_fishing_b = m * growth_a + (1 - m) * growth_b
    marginal_a = sp.diff(growth_a, N_A)
    marginal_b = sp.diff(growth_b, N_B)

    shared_no_take_condition = sp.factor(
        (1 - m) * (marginal_a + marginal_b)
        + (2 * m - 1) * marginal_a * marginal_b
        - 1
    )

    no_take_a_equilibrium = sp.factor(N_A - pre_fishing_a)
    no_take_a_effort_b = sp.factor(1 - N_B / pre_fishing_b)

    no_take_b_equilibrium = sp.factor(N_B - pre_fishing_b)
    no_take_b_effort_a = sp.factor(1 - N_A / pre_fishing_a)

    complete_a_effort_b = sp.factor(1 - 1 / ((1 - m) * sp.sqrt(r_B)))
    complete_b_effort_a = sp.factor(1 - 1 / ((1 - m) * sp.sqrt(r_A)))

    return {
        "e_A = 0": (
            no_take_a_equilibrium,
            shared_no_take_condition,
            no_take_a_effort_b,
        ),
        "e_B = 0": (
            no_take_b_equilibrium,
            shared_no_take_condition,
            no_take_b_effort_a,
        ),
        "e_A = 1": (complete_a_effort_b,),
        "e_B = 1": (complete_b_effort_a,),
    }


def linked_fishing_yield_no_take_lagrange_reduction() -> dict[str, sp.Expr]:
    q_A, q_B = sp.symbols("q_A q_B", positive=True)

    lambda_b_from_effort_derivative = sp.Integer(-1)
    lambda_a_from_population_a_derivative = sp.solve(
        sp.Eq(m * q_A + lambda_A * (1 - (1 - m) * q_A), 0),
        lambda_A,
    )[0]
    population_b_derivative = sp.factor((1 - m) * q_B - m * q_B * lambda_A - 1)
    reduced_population_b_derivative = sp.factor(
        population_b_derivative.subs(
            lambda_A,
            lambda_a_from_population_a_derivative,
        )
    )
    stationarity_condition = sp.factor(
        sp.together(reduced_population_b_derivative).as_numer_denom()[0]
    )
    paper_stationarity_condition = (
        (1 - m) * (q_A + q_B)
        + (2 * m - 1) * q_A * q_B
        - 1
    )

    return {
        "dL/de_B gives lambda_B": lambda_b_from_effort_derivative,
        "dL/dN_A gives lambda_A": lambda_a_from_population_a_derivative,
        "dL/dN_B after lambda_B = -1": population_b_derivative,
        "dL/dN_B after substituting lambda_A": reduced_population_b_derivative,
        "stationarity condition in q_A, q_B": paper_stationarity_condition,
        "stationarity expression check": sp.simplify(
            stationarity_condition - paper_stationarity_condition
        ),
    }


def linked_fishing_yield_no_take_marginal_reduction() -> dict[str, sp.Expr]:
    x, y, a, b = sp.symbols("x y a b", positive=True)

    marginal_condition = sp.factor(
        (1 - m) * (x**2 + y**2) + (2 * m - 1) * x**2 * y**2 - 1
    )
    y_from_x = sp.sqrt((1 - (1 - m) * x**2) / (1 - m + (2 * m - 1) * x**2))
    x_from_y = sp.sqrt((1 - (1 - m) * y**2) / (1 - m + (2 * m - 1) * y**2))

    no_take_a_equation = sp.factor(
        K_A * (a - x) / x
        - (1 - m) * K_A * a * (a - x)
        - m * K_B * b * (b - y)
    )
    no_take_b_equation = sp.factor(
        K_B * (b - y) / y
        - m * K_A * a * (a - x)
        - (1 - m) * K_B * b * (b - y)
    )
    no_take_a_effort = sp.factor(
        1
        - (K_B * (b - y) / y)
        / (m * K_A * a * (a - x) + (1 - m) * K_B * b * (b - y))
    )
    no_take_b_effort = sp.factor(
        1
        - (K_A * (a - x) / x)
        / ((1 - m) * K_A * a * (a - x) + m * K_B * b * (b - y))
    )

    return {
        "marginal_condition": marginal_condition,
        "y_A(x)": y_from_x,
        "x_B(y)": x_from_y,
        "e_A = 0 scalar equation": no_take_a_equation,
        "e_A = 0 effort": no_take_a_effort,
        "e_B = 0 scalar equation": no_take_b_equation,
        "e_B = 0 effort": no_take_b_effort,
    }


def exact_equilibrium_reduction() -> tuple[sp.Expr, sp.Expr, list[sp.Expr]]:
    eq_a, eq_b = equilibrium_equations()
    numerator_a = sp.together(eq_a).as_numer_denom()[0]
    numerator_b = sp.together(eq_b).as_numer_denom()[0]
    n_b_from_n_a = sp.solve(numerator_a, N_B)[0]
    resultant = sp.factor(sp.resultant(numerator_a, numerator_b, N_B))

    cubic = [
        factor
        for factor, _ in sp.factor_list(resultant)[1]
        if sp.degree(factor, N_A) == 3
    ][0]
    coefficients = sp.Poly(cubic, N_A).all_coeffs()

    return sp.factor(n_b_from_n_a), sp.factor(cubic), coefficients


def exact_fishing_equilibrium_reduction() -> tuple[sp.Expr, sp.Expr]:
    eq_a, eq_b = fishing_equilibrium_equations()
    numerator_a = sp.together(eq_a).as_numer_denom()[0]
    numerator_b = sp.together(eq_b).as_numer_denom()[0]
    n_b_from_n_a = sp.solve(numerator_a, N_B)[0]
    resultant = sp.factor(sp.resultant(numerator_a, numerator_b, N_B))

    cubic = [
        factor
        for factor, _ in sp.factor_list(resultant)[1]
        if sp.degree(factor, N_A) == 3
    ][0]

    return sp.factor(n_b_from_n_a), sp.factor(cubic)


def conservation_relations() -> tuple[sp.Expr, sp.Expr]:
    growth_a = beverton_holt_growth(N_A, r_A, K_A)
    growth_b = beverton_holt_growth(N_B, r_B, K_B)
    next_a, next_b = linked_map()

    total_relation = sp.factor(next_a + next_b - (N_A + N_B))
    difference_relation = sp.factor(next_a - next_b - (N_A - N_B))

    return (
        sp.factor(total_relation - (growth_a + growth_b - N_A - N_B)),
        sp.factor(difference_relation - ((1 - 2 * m) * (growth_a - growth_b) - N_A + N_B)),
    )


def main() -> None:
    next_a, next_b = linked_map()
    fishing_next_a, fishing_next_b = linked_map_with_fishing()
    eq_a, eq_b = equilibrium_equations()
    jacobian = jacobian_at_extinction((next_a, next_b))
    jacobian_trace, jacobian_determinant = trace_and_determinant(jacobian)
    jacobian_perron_eigenvalue = perron_eigenvalue(jacobian)
    fishing_jacobian = jacobian_at_extinction((fishing_next_a, fishing_next_b))
    fishing_trace, fishing_determinant = trace_and_determinant(fishing_jacobian)
    jury_conditions = jury_stability_conditions(fishing_jacobian)
    fishing_boundary = extinction_boundary_expression(fishing_jacobian)
    fishing_boundary_e_b = solve_extinction_boundary_for_e_b(fishing_jacobian)
    fishing_boundary_e_a = solve_extinction_boundary_for_e_a(fishing_jacobian)
    fishing_yield = linked_fishing_yield_expression()
    yield_lagrangian, yield_lagrange_equations = linked_fishing_yield_lagrange_system()
    reduced_yield_conditions = linked_fishing_yield_reduced_interior_conditions()
    interior_candidate = linked_fishing_yield_interior_candidate()
    boundary_candidates = linked_fishing_yield_boundary_conditions()
    no_take_lagrange_reduction = linked_fishing_yield_no_take_lagrange_reduction()
    no_take_marginal_reduction = linked_fishing_yield_no_take_marginal_reduction()
    n_b_from_n_a, exact_cubic, exact_cubic_coefficients = exact_equilibrium_reduction()
    fishing_n_b_from_n_a, fishing_exact_cubic = exact_fishing_equilibrium_reduction()

    print("Linked Beverton-Holt map")
    print("========================")
    print(f"N_A(t+1) = {sp.latex(next_a)}")
    print(f"N_B(t+1) = {sp.latex(next_b)}")
    print()

    print("Equilibrium equations")
    print("=====================")
    print(f"0 = {sp.latex(eq_a)}")
    print(f"0 = {sp.latex(eq_b)}")
    print()

    print("Extinction equilibrium")
    print("======================")
    print("(N_A*, N_B*) = (0, 0)")
    print()

    print("When m = 0")
    print("==========")
    print(f"N_A* = {sp.latex(K_A * (r_A - 1))}")
    print(f"N_B* = {sp.latex(K_B * (r_B - 1))}")
    print()

    print("Exact equilibrium reduction")
    print("===========================")
    print("N_A* is a root of P(N_A)=0, where P is cubic.")
    print(f"N_B*(N_A*) = {sp.latex(n_b_from_n_a)}")
    print(f"P(N_A) = {sp.latex(exact_cubic)}")
    print("Cubic coefficients:")
    for index, coefficient in enumerate(exact_cubic_coefficients):
        power = len(exact_cubic_coefficients) - index - 1
        print(f"N_A^{power}: {sp.latex(coefficient)}")
    print()

    print("Jacobian at extinction")
    print("======================")
    print(sp.latex(jacobian))
    print(f"trace(M) = {sp.latex(jacobian_trace)}")
    print(f"det(M) = {sp.latex(jacobian_determinant)}")
    print(f"rho(M) = {sp.latex(jacobian_perron_eigenvalue)}")
    print()

    print("Jacobian at extinction with fishing")
    print("===================================")
    print(sp.latex(fishing_jacobian))
    print()

    print("Exact fishing equilibrium reduction")
    print("===================================")
    print("N_A* is again a root of a cubic P_f(N_A)=0.")
    print(f"N_B*(N_A*) = {sp.latex(fishing_n_b_from_n_a)}")
    print(f"degree P_f = {sp.degree(fishing_exact_cubic, N_A)}")
    print()

    print("Fishing extinction boundary det(I-M) = 0")
    print("========================================")
    print(f"trace(M) = {sp.latex(fishing_trace)}")
    print(f"det(M) = {sp.latex(fishing_determinant)}")
    print()
    print("Jury stability conditions for both eigenvalues in the unit disk:")
    print(f"1 - trace(M) + det(M) > 0: {sp.latex(jury_conditions[0])}")
    print(f"1 + trace(M) + det(M) > 0: {sp.latex(jury_conditions[1])}")
    print(f"1 - det(M) > 0: {sp.latex(jury_conditions[2])}")
    print()
    print(f"det(I-M) = {sp.latex(fishing_boundary)}")
    print(f"e_B = {sp.latex(fishing_boundary_e_b)}")
    print(f"e_A = {sp.latex(fishing_boundary_e_a)}")
    print()

    print("Linked fishing yield maximization")
    print("=================================")
    print(f"Y = {sp.latex(fishing_yield)}")
    print(f"L = {sp.latex(yield_lagrangian)}")
    print("First-order equations:")
    for equation in yield_lagrange_equations:
        print(f"0 = {sp.latex(equation)}")
    print()
    print("Reduced interior yield conditions:")
    for equation in reduced_yield_conditions:
        print(f"0 = {sp.latex(equation)}")
    print("Interior candidate:")
    print(f"N_A = {sp.latex(interior_candidate[0])}")
    print(f"N_B = {sp.latex(interior_candidate[1])}")
    print(f"e_A = {sp.latex(interior_candidate[2])}")
    print(f"e_B = {sp.latex(interior_candidate[3])}")
    print()
    print("Boundary yield candidate conditions:")
    for boundary_name, equations in boundary_candidates.items():
        print(boundary_name)
        for equation in equations:
            print(f"0 = {sp.latex(equation)}")
    print()
    print("No-take edge Lagrange reduction:")
    for name, expression in no_take_lagrange_reduction.items():
        print(f"{name}: {sp.latex(expression)}")
    print()
    print("No-take edge marginal-derivative reduction:")
    for name, expression in no_take_marginal_reduction.items():
        print(f"{name}: {sp.latex(expression)}")


if __name__ == "__main__":
    main()
