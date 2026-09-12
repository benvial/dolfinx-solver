"""Judging the answers the smoke test knows in advance, for either variant."""

import math

import pytest

from wheelbuild import petsc
from wheeltest import smoke


def test_the_plane_wave_is_the_exact_solution_of_the_helmholtz_problem():
    """Everything the complex variant's check asserts rests on this."""
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


def test_the_sine_product_solves_the_poisson_problem_with_that_source():
    """The real variant's problem: -Δu = 2π²u for u = sin(πx)sin(πy)."""
    step = 1e-4

    def exact(x, y):
        return math.sin(math.pi * x) * math.sin(math.pi * y)

    point = (0.3, 0.4)
    laplacian = sum(
        (
            exact(*[c + step * (axis == index) for index, c in enumerate(point)])
            - 2 * exact(*point)
            + exact(*[c - step * (axis == index) for index, c in enumerate(point)])
        )
        / step**2
        for axis in (0, 1)
    )

    residual = -laplacian - 2 * math.pi**2 * exact(*point)

    assert abs(residual) < 1e-3


@pytest.mark.parametrize("variant", ["complex", "real"])
def test_a_solution_on_the_exact_one_passes(variant):
    assert smoke.error_problem(1e-5, variant=variant) is None


@pytest.mark.parametrize("variant", ["complex", "real"])
def test_a_solution_that_is_merely_near_the_exact_one_fails(variant):
    problem = smoke.error_problem(0.2, variant=variant)

    assert problem is not None
    assert "not discretisation error" in problem
    assert smoke.PROBLEMS[variant].name in problem


def test_the_complex_variant_refuses_a_real_solution_vector():
    problem = smoke.dtype_problem("float64", variant="complex")

    assert problem is not None
    assert "complex scalars" in problem


def test_the_real_variant_refuses_a_complex_solution_vector():
    """The wrong-way failure: `dolfinx-solver-real` must not compute in C."""
    problem = smoke.dtype_problem("complex128", variant="real")

    assert problem is not None
    assert "real scalars" in problem


@pytest.mark.parametrize(
    ("variant", "dtype"), [("complex", "complex128"), ("real", "float64")]
)
def test_each_variants_own_solution_vector_passes(variant, dtype):
    assert smoke.dtype_problem(dtype, variant=variant) is None


def test_the_variant_defaults_to_the_one_the_build_declares():
    """One declaration (spec §6), so the stage cannot drift from the build."""
    expected = smoke.PROBLEMS[petsc.SCALAR_TYPE]

    assert smoke.problem_for() is expected
    assert smoke.dtype_problem(f"{expected.dtype_prefix}128") is None


def test_a_variant_this_build_has_no_name_for_is_refused():
    """`variant="Complex"` would otherwise silently pick no problem at all."""
    with pytest.raises(ValueError, match="Complex"):
        smoke.problem_for("Complex")


def test_every_variant_the_build_knows_has_a_problem_to_solve():
    assert set(smoke.PROBLEMS) == set(petsc.SCALAR_TYPES)


@pytest.mark.parametrize("variant", ["complex", "real"])
def test_what_the_stage_proves_names_the_variants_own_problem(variant):
    line = smoke.proves(variant)

    assert smoke.PROBLEMS[variant].name in line
    assert "SLEPc" in line


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
