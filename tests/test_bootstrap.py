from pathlib import Path

import pytest

from dolfinx_solver import _bootstrap

SITE_PACKAGES = Path("/venv/lib/python3.12/site-packages")
PACKAGE_DIR = SITE_PACKAGES / "dolfinx_solver"


class FakeMPI:
    """Stand-in for ``mpi4py.MPI``, which needs a real libmpi to import."""


def _importer(available: dict[str, object]):
    def import_module(name: str) -> object:
        try:
            return available[name]
        except KeyError:
            raise ImportError(f"No module named {name!r}") from None

    return import_module


def test_a_petsc4py_beside_this_package_is_the_one_the_wheel_ships():
    origin = SITE_PACKAGES / "petsc4py" / "__init__.py"

    assert _bootstrap.foreign_petsc4py_problem(origin, PACKAGE_DIR) is None


def test_no_petsc4py_at_all_is_not_a_conflict():
    assert _bootstrap.foreign_petsc4py_problem(None, PACKAGE_DIR) is None


def test_a_petsc4py_from_elsewhere_on_the_path_is_reported():
    origin = Path("/usr/lib/python3/dist-packages/petsc4py/__init__.py")

    problem = _bootstrap.foreign_petsc4py_problem(origin, PACKAGE_DIR)

    assert problem is not None
    assert str(origin) in problem
    assert "petsc4py" in problem


def test_a_namespace_petsc4py_without_a_file_is_reported():
    """A petsc4py whose origin cannot be located is not ours either."""
    problem = _bootstrap.foreign_petsc4py_problem(Path("built-in"), PACKAGE_DIR)

    assert problem is not None


def test_mpi_is_imported_before_anything_compiled():
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return FakeMPI()

    _bootstrap.bootstrap(
        import_module=import_module,
        petsc4py_origin=lambda: None,
        package_dir=PACKAGE_DIR,
        installed_distribution_names=list,
    )

    assert imported[0] == "mpi4py.MPI"


def test_a_foreign_petsc4py_fails_before_mpi_is_touched():
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return FakeMPI()

    with pytest.raises(ImportError) as caught:
        _bootstrap.bootstrap(
            import_module=import_module,
            petsc4py_origin=lambda: Path("/usr/lib/python3/petsc4py/__init__.py"),
            package_dir=PACKAGE_DIR,
        )

    assert "petsc4py" in str(caught.value)
    assert imported == []


def test_a_missing_mpi4py_names_itself_in_the_failure():
    with pytest.raises(ImportError) as caught:
        _bootstrap.bootstrap(
            import_module=_importer({}),
            petsc4py_origin=lambda: None,
            package_dir=PACKAGE_DIR,
            installed_distribution_names=list,
        )

    assert "mpi4py" in str(caught.value)


def test_an_installed_petsc4py_distribution_is_the_overwrite_case():
    """`pip install petsc4py` lands on our files, so the path check misses it."""
    problem = _bootstrap.conflicting_distribution_problem(["numpy", "petsc4py"])

    assert problem is not None
    assert "petsc4py" in problem


def test_every_pypi_petsc_project_is_refused():
    problem = _bootstrap.conflicting_distribution_problem(
        ["petsc", "petsc4py", "slepc", "slepc4py"]
    )

    assert problem is not None
    for name in ("petsc", "petsc4py", "slepc", "slepc4py"):
        assert name in problem


def test_an_environment_without_them_is_accepted():
    assert _bootstrap.conflicting_distribution_problem(["numpy", "mpi4py"]) is None


def test_distribution_names_are_matched_in_normalised_form():
    """PyPI treats PETSc4py, petsc-4py and petsc4py as one project name."""
    assert _bootstrap.canonical_name("PETSc4py") == "petsc4py"
    assert _bootstrap.conflicting_distribution_problem(["PETSc4py"]) is None
    assert (
        _bootstrap.conflicting_distribution_problem(
            _bootstrap.canonical_name(name) for name in ["PETSc4py"]
        )
        is not None
    )


def test_a_foreign_distribution_fails_before_mpi_is_touched():
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return FakeMPI()

    with pytest.raises(ImportError) as caught:
        _bootstrap.bootstrap(
            import_module=import_module,
            petsc4py_origin=lambda: None,
            package_dir=PACKAGE_DIR,
            installed_distribution_names=lambda: ["petsc4py"],
        )

    assert "petsc4py" in str(caught.value)
    assert imported == []


def test_this_environment_carries_no_conflicting_distribution():
    """The dev environment has to be one the wheel would import cleanly in."""
    names = _bootstrap.installed_distribution_names()

    assert _bootstrap.conflicting_distribution_problem(names) is None


def test_the_real_package_runs_the_bootstrap_on_import():
    """Importing the package must have ordered MPI ahead of any binding."""
    import sys

    import dolfinx_solver

    assert dolfinx_solver.mpi_module() is sys.modules["mpi4py.MPI"]
