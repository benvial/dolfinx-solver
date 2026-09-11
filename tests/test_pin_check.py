"""The declared mpich bound and the MPICH this build vendors are one number.

Spec §10 and ADR-0001: `libmpifort` is built against one MPICH series'
exported symbols, so the `mpich` requirement in `pyproject.toml` has to name
that series exactly — not a floor that admits an older runtime, and not a
newer series than the one we built against.
"""

import pytest

from wheelbuild import mpich, pin_check

PYPROJECT = """\
[project]
name = "dolfinx-solver-complex"
dependencies = [
  "fenics-basix>=0.11.0,<0.12.0",
  "mpi4py",
  "mpich>=5.0,<6",
  "numpy>=2"
]
"""


def test_the_declared_requirement_is_read_out_of_the_dependencies():
    assert pin_check.declared_mpich_requirement(PYPROJECT) == "mpich>=5.0,<6"


def test_a_pyproject_without_an_mpich_dependency_is_an_error():
    without = PYPROJECT.replace('  "mpich>=5.0,<6",\n', "")

    with pytest.raises(LookupError, match="mpich"):
        pin_check.declared_mpich_requirement(without)


def test_a_series_pin_is_a_floor_and_a_ceiling():
    assert pin_check.expected_specifier("5.0.1") == ">=5.0,<6"
    assert pin_check.expected_specifier("4.2.3") == ">=4.0,<5"


def test_the_declared_bound_and_the_vendored_build_agree():
    assert pin_check.pin_problem("mpich>=5.0,<6", "5.0.1") is None


def test_the_specifiers_may_be_written_in_any_order():
    assert pin_check.pin_problem("mpich<6,>=5.0", "5.0.1") is None


def test_a_bump_of_the_vendored_build_past_the_bound_is_reported():
    problem = pin_check.pin_problem("mpich>=5.0,<6", "6.0.0")

    assert problem is not None
    assert "6.0.0" in problem


def test_a_loose_floor_is_reported_even_though_the_build_satisfies_it():
    """`mpich>=4.2` admits a libmpi older than the libmpifort we vendor."""
    problem = pin_check.pin_problem("mpich>=4.2", "5.0.1")

    assert problem is not None
    assert ">=5.0,<6" in problem


def test_an_exact_pin_is_reported_too():
    """Patch releases of the wheel are routine and ABI-compatible (ADR-0001)."""
    assert pin_check.pin_problem("mpich==5.0.1", "5.0.1") is not None


def test_a_requirement_for_something_other_than_mpich_is_reported():
    assert pin_check.pin_problem("openmpi>=5.0,<6", "5.0.1") is not None


def test_this_repository_passes_its_own_check():
    """The bound in pyproject.toml and wheelbuild's MPICH driver, for real."""
    declared = pin_check.declared_mpich_requirement(
        pin_check.PYPROJECT_PATH.read_text(encoding="utf-8")
    )

    assert pin_check.pin_problem(declared, mpich.MPICH_VERSION) is None
