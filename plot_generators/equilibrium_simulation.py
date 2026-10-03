"""Numerical sustainable catches from the linked Beverton--Holt update.

The solver brackets the limiting equilibrium with monotone population
iterations. Safeguarded Newton steps accelerate the upper iteration near
critical growth, where ordinary time stepping can retain a long transient.
No extinction boundary or harvesting-region formula is used to classify an
effort pair. Convergence is measured by the remaining catch interval, rather
than by a small change between successive population steps.
"""
from __future__ import annotations

import numpy as np


def equilibrium_yields(
    effort_a,
    effort_b,
    *,
    r_a,
    r_b,
    m: float,
    k_a: float,
    k_b: float,
    yield_atol: float = 1e-12,
    yield_rtol: float = 1e-9,
    max_iterations: int = 200,
) -> np.ndarray:
    """Return broadcast catches at the equilibrium reached from positive starts.

    ``effort_a``, ``effort_b``, ``r_a`` and ``r_b`` may be broadcast arrays.
    Each returned catch has an upper/lower numerical bracket of width at most
    ``yield_atol + yield_rtol * upper_catch``. Catches whose upper bound is at
    most ``yield_atol`` are returned as zero. The bounds hold up to floating
    point rounding; an exhausted iteration budget raises ``RuntimeError``.

    For the monotone BH map T, the population ceiling U_i=sum_j M_ij K_j
    satisfies T(U)<=U and bounds the greatest nonnegative equilibrium N*.
    For h(c)=T(cU)/c, convexity gives h(c)>=T(U)+(1-c)D, where
    D_i=sum_j M_ij U_j (U_j/K_j)/(1+U_j/K_j)^2. Thus choosing
    c=max(0,min(1,1-max_i[(U_i-T_i(U))/D_i])) makes L=cU a
    subsolution: T(L)>=L. Zero rows impose no restriction. Previous
    lower subsolutions are retained and advanced by T. Hence L<=N*<=U,
    and monotonicity of the catch function supplies the stopping interval.

    Concavity of T makes the Newton update for U-T(U)=0 another decreasing
    supersolution when I-T'(U) is a nonsingular M-matrix. Both that matrix
    condition and the candidate bounds are checked before accepting a step;
    otherwise the solver takes the ordinary population update T(U).
    """
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not (np.isfinite(m) and 0 <= m <= 1):
        raise ValueError("m must be finite and between zero and one")
    if not (np.isfinite(k_a) and k_a > 0 and np.isfinite(k_b) and k_b > 0):
        raise ValueError("population scales must be finite and positive")
    if not (np.isfinite(yield_atol) and yield_atol > 0
            and np.isfinite(yield_rtol) and yield_rtol >= 0):
        raise ValueError("yield_atol must be positive and yield_rtol nonnegative")

    e_a, e_b, r_a, r_b = np.broadcast_arrays(
        np.asarray(effort_a, dtype=float), np.asarray(effort_b, dtype=float),
        np.asarray(r_a, dtype=float), np.asarray(r_b, dtype=float),
    )
    if not all(np.all(np.isfinite(value)) for value in (e_a, e_b, r_a, r_b)):
        raise ValueError("efforts and growth rates must be finite")
    if np.any((e_a < 0) | (e_a > 1) | (e_b < 0) | (e_b > 1)):
        raise ValueError("efforts must lie between zero and one")
    if np.any((r_a < 0) | (r_b < 0)):
        raise ValueError("growth rates must be nonnegative")

    q = 1 - m
    s_a, s_b = 1 - e_a, 1 - e_b
    upper_a = s_a * (q * r_a * k_a + m * r_b * k_b)
    upper_b = s_b * (m * r_a * k_a + q * r_b * k_b)
    if not (np.all(np.isfinite(upper_a)) and np.all(np.isfinite(upper_b))):
        raise ValueError("parameter magnitudes overflow the population ceiling")

    def update(population_a, population_b):
        growth_a = k_a * r_a * population_a / (k_a + population_a)
        growth_b = k_b * r_b * population_b / (k_b + population_b)
        preharvest_a = q * growth_a + m * growth_b
        preharvest_b = m * growth_a + q * growth_b
        catch = e_a * preharvest_a + e_b * preharvest_b
        return s_a * preharvest_a, s_b * preharvest_b, catch

    lower_a = np.zeros_like(upper_a)
    lower_b = np.zeros_like(upper_b)
    for iteration in range(max_iterations + 1):
        next_a, next_b, upper_catch = update(upper_a, upper_b)
        residual_a = np.maximum(upper_a - next_a, 0.)
        residual_b = np.maximum(upper_b - next_b, 0.)
        density_a, density_b = upper_a / k_a, upper_b / k_b
        curvature_a = r_a * upper_a * density_a / (1 + density_a) ** 2
        curvature_b = r_b * upper_b * density_b / (1 + density_b) ** 2
        lower_slope_a = s_a * (q * curvature_a + m * curvature_b)
        lower_slope_b = s_b * (m * curvature_a + q * curvature_b)
        # Form D directly: T(U)-T'(U)U would lose accuracy through
        # cancellation near critical growth or a nearly empty patch.
        fraction_a = np.divide(
            residual_a, lower_slope_a,
            out=np.where(residual_a > 0, np.inf, 0.), where=lower_slope_a > 0,
        )
        fraction_b = np.divide(
            residual_b, lower_slope_b,
            out=np.where(residual_b > 0, np.inf, 0.), where=lower_slope_b > 0,
        )
        scale = np.clip(1 - np.maximum(fraction_a, fraction_b), 0., 1.)
        lower_a = np.maximum(lower_a, scale * upper_a)
        lower_b = np.maximum(lower_b, scale * upper_b)
        lower_a, lower_b, _ = update(lower_a, lower_b)
        # These clips only remove floating point crossings of equal bounds.
        lower_a = np.minimum(lower_a, upper_a)
        lower_b = np.minimum(lower_b, upper_b)
        _, _, lower_catch = update(lower_a, lower_b)
        lower_catch = np.minimum(lower_catch, upper_catch)
        width = upper_catch - lower_catch
        unresolved = width > yield_atol + yield_rtol * upper_catch
        if not np.any(unresolved):
            return np.asarray(np.where(
                upper_catch <= yield_atol, 0., (lower_catch + upper_catch) / 2,
            ))
        if iteration == max_iterations:
            worst = int(np.argmax(np.where(unresolved, width, -np.inf)))
            raise RuntimeError(
                f"equilibrium catch did not converge for {np.count_nonzero(unresolved)} "
                f"effort pairs after {max_iterations} iterations "
                f"(largest unresolved catch interval {np.max(width[unresolved]):.3g}; "
                f"r_a={r_a.flat[worst]:.17g}, r_b={r_b.flat[worst]:.17g}, "
                f"effort_a={e_a.flat[worst]:.17g}, effort_b={e_b.flat[worst]:.17g})"
            )

        derivative_a = r_a / (1 + upper_a / k_a) ** 2
        derivative_b = r_b / (1 + upper_b / k_b) ** 2
        # Entries of I-T'(U); positive diagonals/determinant guarantee its
        # inverse is nonnegative because its off-diagonal entries are <=0.
        a = 1 - s_a * q * derivative_a
        b = -s_a * m * derivative_b
        c = -s_b * m * derivative_a
        d = 1 - s_b * q * derivative_b
        determinant = a * d - b * c
        residual_a, residual_b = upper_a - next_a, upper_b - next_b
        cushion = 64 * np.finfo(float).eps * np.maximum(
            1., np.maximum(upper_a, upper_b),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate_a = upper_a - (d * residual_a - b * residual_b) / determinant
            candidate_b = upper_b - (a * residual_b - c * residual_a) / determinant
        finite_nonnegative = (
            np.isfinite(candidate_a) & np.isfinite(candidate_b)
            & (candidate_a >= -cushion) & (candidate_b >= -cushion)
        )
        candidate_a = np.where(np.isfinite(candidate_a), candidate_a, upper_a)
        candidate_b = np.where(np.isfinite(candidate_b), candidate_b, upper_b)
        # Clipping tiny negative roundoff also keeps rejected candidates out
        # of the growth functions' poles during the safeguard evaluation.
        candidate_a = np.maximum(candidate_a, 0.)
        candidate_b = np.maximum(candidate_b, 0.)
        candidate_next_a, candidate_next_b, _ = update(candidate_a, candidate_b)
        safe = (
            finite_nonnegative & (determinant > 0) & (a > 0) & (d > 0)
            & (candidate_a >= lower_a - cushion)
            & (candidate_b >= lower_b - cushion)
            & (candidate_a <= upper_a + cushion)
            & (candidate_b <= upper_b + cushion)
            & (candidate_next_a <= candidate_a + cushion)
            & (candidate_next_b <= candidate_b + cushion)
        )
        candidate_a = np.minimum(candidate_a, upper_a)
        candidate_b = np.minimum(candidate_b, upper_b)
        upper_a = np.where(unresolved, np.where(safe, candidate_a, next_a), upper_a)
        upper_b = np.where(unresolved, np.where(safe, candidate_b, next_b), upper_b)

    raise AssertionError("unreachable iteration exit")
