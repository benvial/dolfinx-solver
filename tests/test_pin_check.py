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


PYPROJECT_WITH_TRIO = """\
[project]
name = "dolfinx-solver-complex"
dependencies = [
  "cffi",
  "fenics-basix>=0.11.0,<0.12.0",
  "fenics-ffcx>=0.11.0,<0.12.0",
  "fenics-ufl>=2026.1.0,<2026.2.0",
  "mpi4py",
  "mpich>=5.0,<6",
  "numpy>=2"
]
"""


UPSTREAM_PYPROJECT = """\
[project]
name = "fenics-dolfinx"
version = "0.11.0.post0"
dependencies = [
      "numpy>=2",
      "cffi",
      "mpi4py",
      "fenics-basix>=0.11.0,<0.12.0",
      "fenics-ffcx>=0.11.0,<0.12.0",
      "fenics-ufl>=2026.1.0,<2026.2.0",
]
"""


def test_the_upstream_pyproject_is_read_at_the_mirrored_tag():
    url = pin_check.upstream_pyproject_url("0.11.0.post0")

    assert url.endswith("/v0.11.0.post0/python/pyproject.toml")
    assert url.startswith("https://")


def test_our_trio_pins_and_upstreams_agree():
    assert pin_check.trio_problem(PYPROJECT_WITH_TRIO, UPSTREAM_PYPROJECT) is None


def test_the_specifiers_may_be_written_in_any_order_upstream():
    reordered = UPSTREAM_PYPROJECT.replace(
        '"fenics-ufl>=2026.1.0,<2026.2.0"', '"fenics-ufl<2026.2.0,>=2026.1.0"'
    )

    assert pin_check.trio_problem(PYPROJECT_WITH_TRIO, reordered) is None


def test_a_trio_pin_that_drifts_from_upstreams_is_reported():
    """Upstream moved to the next basix series; our metadata did not."""
    drifted = UPSTREAM_PYPROJECT.replace(
        '"fenics-basix>=0.11.0,<0.12.0"', '"fenics-basix>=0.12.0,<0.13.0"'
    )

    problem = pin_check.trio_problem(PYPROJECT_WITH_TRIO, drifted)

    assert problem is not None
    assert "fenics-basix" in problem
    assert ">=0.12.0,<0.13.0" in problem


def test_widening_our_own_pin_is_reported_even_though_upstreams_is_admitted():
    """`fenics-ffcx>=0.11.0` admits a release upstream's own pin excludes."""
    widened = PYPROJECT_WITH_TRIO.replace(
        '"fenics-ffcx>=0.11.0,<0.12.0"', '"fenics-ffcx>=0.11.0"'
    )

    problem = pin_check.trio_problem(widened, UPSTREAM_PYPROJECT)

    assert problem is not None
    assert "fenics-ffcx" in problem


def test_a_trio_member_we_do_not_declare_at_all_is_reported():
    without = PYPROJECT_WITH_TRIO.replace('  "fenics-ufl>=2026.1.0,<2026.2.0",\n', "")

    problem = pin_check.trio_problem(without, UPSTREAM_PYPROJECT)

    assert problem is not None
    assert "fenics-ufl" in problem


def test_a_trio_member_upstream_stops_declaring_is_reported():
    """A tag whose pyproject we cannot read pins nothing; that is the failure."""
    without = UPSTREAM_PYPROJECT.replace(
        '      "fenics-ufl>=2026.1.0,<2026.2.0",\n', ""
    )

    problem = pin_check.trio_problem(PYPROJECT_WITH_TRIO, without)

    assert problem is not None
    assert "fenics-ufl" in problem


def test_this_repository_pins_the_trio_as_this_file_records_upstream_pinning_it():
    """Our real pyproject.toml against a recorded copy of upstream's.

    The recorded copy is what makes this runnable offline, and it is also its
    limit: it cannot notice that upstream has since changed. Only CI's
    `pin_check --fetch-upstream` reads the live file, and that is the check
    that catches drift — this one catches our own pins moving away from what
    upstream declared when the release was mirrored.
    """
    ours = pin_check.PYPROJECT_PATH.read_text(encoding="utf-8")

    assert pin_check.trio_problem(ours, UPSTREAM_PYPROJECT) is None


def test_an_upstream_file_for_another_release_is_not_compared_against():
    """A stale unpacked source tree is exactly what `--upstream-pyproject` hits."""
    older = UPSTREAM_PYPROJECT.replace('version = "0.11.0.post0"', 'version = "0.10.0"')

    problem = pin_check.trio_problem(PYPROJECT_WITH_TRIO, older)

    assert problem is not None
    assert "0.10.0" in problem


def test_metadata_for_another_project_entirely_is_reported():
    ours = PYPROJECT_WITH_TRIO

    problem = pin_check.trio_problem(ours, ours)

    assert problem is not None
    assert "fenics-dolfinx" in problem


def test_a_pin_upstream_leaves_unconditional_may_not_gain_a_marker():
    """A marker changes what pip installs as surely as a specifier does."""
    conditional = PYPROJECT_WITH_TRIO.replace(
        '"fenics-ffcx>=0.11.0,<0.12.0"',
        "\"fenics-ffcx>=0.11.0,<0.12.0 ; python_version < '3.13'\"",
    )

    problem = pin_check.trio_problem(conditional, UPSTREAM_PYPROJECT)

    assert problem is not None
    assert "fenics-ffcx" in problem


def test_a_pin_upstream_leaves_bare_may_not_gain_an_extra():
    with_extra = PYPROJECT_WITH_TRIO.replace(
        '"fenics-basix>=0.11.0,<0.12.0"', '"fenics-basix[docs]>=0.11.0,<0.12.0"'
    )

    problem = pin_check.trio_problem(with_extra, UPSTREAM_PYPROJECT)

    assert problem is not None
    assert "fenics-basix" in problem
