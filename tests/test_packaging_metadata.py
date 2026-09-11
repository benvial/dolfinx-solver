"""What the two published distributions promise, checked without a build.

The wheel itself takes hours to produce, so the metadata rules that a bad
release would break — the meta-package pinning its binary sibling exactly, the
runtime never naming the sdist-only PyPI PETSc projects — are asserted against
the ``pyproject.toml`` files directly.
"""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from wheelbuild.mpich import MPICH_VERSION
from wheelbuild.pin_check import expected_specifier
from wheelbuild.version import __version__

#: MPICH release the wheel vendors libmpifort from, and the series the runtime
#: dependency is therefore bounded to (ADR-0001). Both come from the build
#: driver that owns them, so the metadata is checked against what the wheel is
#: actually built from rather than against a literal copied beside it.
MPICH_SERIES = MPICH_VERSION
MPICH_REQUIREMENT = expected_specifier(MPICH_VERSION)

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY_PROJECT = REPO_ROOT / "pyproject.toml"
META_PROJECT = REPO_ROOT / "meta" / "pyproject.toml"

#: Projects that ship PETSc or SLEPc as source-only PyPI sdists. Depending on
#: any of them would make an install compile PETSc, or load a second one next
#: to the vendored build. Spec §10.
FORBIDDEN_DEPENDENCIES = ("petsc", "petsc4py", "slepc", "slepc4py")


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text())["project"]


@pytest.fixture(scope="module")
def binary() -> dict:
    return _project(BINARY_PROJECT)


@pytest.fixture(scope="module")
def meta() -> dict:
    return _project(META_PROJECT)


def test_the_binary_distribution_is_the_complex_variant(binary):
    assert binary["name"] == "dolfinx-solver-complex"


def test_the_binary_distribution_reads_its_version_from_the_module(binary):
    """One source of truth: ``dolfinx_solver.__version__`` (spec §3)."""
    assert "version" in binary["dynamic"]
    assert "version" not in binary


def test_the_binary_distribution_keeps_the_abi3_python_floor(binary):
    assert binary["requires-python"] == ">=3.12"


def test_the_binary_distribution_is_lgpl(binary):
    assert binary["license"] == "LGPL-3.0-or-later"


def test_the_upstream_trio_is_pinned_as_upstream_pins_it(binary):
    """Verbatim from DOLFINx 0.11.0.post0's own python/pyproject.toml."""
    assert "fenics-basix>=0.11.0,<0.12.0" in binary["dependencies"]
    assert "fenics-ffcx>=0.11.0,<0.12.0" in binary["dependencies"]
    assert "fenics-ufl>=2026.1.0,<2026.2.0" in binary["dependencies"]


def test_the_mpi_dependencies_match_the_shared_process_contract(binary):
    assert f"mpich{MPICH_REQUIREMENT}" in binary["dependencies"]
    assert "mpi4py" in binary["dependencies"]


def test_the_mpich_pin_bounds_a_series_rather_than_setting_a_floor():
    """ADR-0001: the vendored libmpifort is built against one series' libmpi.

    A floor alone resolves to whatever the wheel publishes next, which can be
    an older runtime than the Fortran layer we ship against it — and that
    fails at the user's first import, not at install.
    """
    specifier = SpecifierSet(MPICH_REQUIREMENT)

    assert Version(MPICH_SERIES) in specifier
    assert Version("4.3.2") not in specifier
    assert Version("6.0.0") not in specifier


@pytest.mark.parametrize("forbidden", FORBIDDEN_DEPENDENCIES)
def test_no_pypi_petsc_project_is_ever_a_dependency(binary, meta, forbidden):
    for dependencies in (binary["dependencies"], meta["dependencies"]):
        assert forbidden not in {_project_name(entry) for entry in dependencies}


@pytest.mark.parametrize(
    "spelling",
    [
        "petsc4py",
        "petsc4py>=3.20",
        "petsc4py~=3.20",
        "petsc4py[opt]==3.20",
        "PETSc4Py ; python_version >= '3.12'",
    ],
)
def test_the_forbidden_name_check_sees_through_every_spelling(spelling):
    """The guard is only worth having if a real requirement line trips it."""
    assert _project_name(spelling) in FORBIDDEN_DEPENDENCIES


def _project_name(requirement: str) -> str:
    """Return the PyPI project a requirement names, normalised (PEP 503)."""
    return canonicalize_name(Requirement(requirement).name)


def test_the_meta_package_holds_the_bare_name(meta):
    assert meta["name"] == "dolfinx-solver"


def test_the_meta_package_pins_the_variant_it_defaults_to(meta):
    assert meta["dependencies"] == [f"dolfinx-solver-complex=={__version__}"]


def test_the_meta_package_releases_in_lockstep(meta):
    assert meta["version"] == __version__


def test_the_meta_package_ships_no_code_of_its_own():
    """It exists to protect the name and pick a default, nothing more."""
    assert not list((REPO_ROOT / "meta").rglob("*.py"))
