"""Judging the two answers the smoke test knows in advance."""

import math

from wheeltest import smoke


def test_the_plane_wave_is_the_exact_solution_of_the_helmholtz_problem():
    """Everything the check asserts rests on this being true."""
    wave_number = smoke.WAVE_NUMBER
    step = 1e-4
    # -u'' - k²u = 0 for u = exp(ikx), read off a second difference.
    values = [
        complex(math.cos(wave_number * x), math.sin(wave_number * x))
        for x in (0.5 - step, 0.5, 0.5 + step)
    ]
    second_derivative = (values[0] - 2 * values[1] + values[2]) / step**2

    residual = -second_derivative - wave_number**2 * values[1]

    assert abs(residual) < 1e-3


def test_a_solution_on_the_plane_wave_passes():
    assert smoke.error_problem(1e-5) is None


def test_a_solution_that_is_merely_near_the_plane_wave_fails():
    problem = smoke.error_problem(0.2)

    assert problem is not None
    assert "not discretisation error" in problem


def test_a_real_solution_vector_is_the_wrong_distribution():
    problem = smoke.dtype_problem("float64")

    assert problem is not None
    assert "complex scalars" in problem


def test_a_complex_solution_vector_is_what_this_wheel_ships():
    assert smoke.dtype_problem("complex128") is None


def test_the_dirichlet_laplacians_own_eigenvalues_pass():
    assert smoke.eigenvalue_problem(smoke.EXPECTED_EIGENVALUES) is None


def test_the_discretisations_own_overestimate_passes():
    """P1 on this mesh overshoots by about half a percent, from above."""
    computed = [value * 1.005 for value in smoke.EXPECTED_EIGENVALUES]

    assert smoke.eigenvalue_problem(computed) is None


def test_an_eigenvalue_in_the_wrong_place_fails():
    computed = [1.0, *smoke.EXPECTED_EIGENVALUES[1:]]

    problem = smoke.eigenvalue_problem(computed)

    assert problem is not None
    assert "eigenvalue 0" in problem


def test_a_self_adjoint_problem_with_a_complex_eigenvalue_fails():
    """A Hermitian pair assembled wrongly shows up here rather than in the norms."""
    computed = [complex(value, value * 1e-3) for value in smoke.EXPECTED_EIGENVALUES]

    problem = smoke.eigenvalue_problem(computed)

    assert problem is not None
    assert "not real" in problem


def test_an_eigensolve_that_converged_on_too_little_names_the_factorisation():
    problem = smoke.eigenvalue_problem(smoke.EXPECTED_EIGENVALUES[:1])

    assert problem is not None
    assert "factorisation" in problem
