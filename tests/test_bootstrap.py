from pathlib import Path

import pytest
from bootstrap_shim import bootstrap_module as _bootstrap

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


def test_the_variant_comes_from_the_distribution_that_shipped_the_payload():
    """`wheelbuild` is not inside the wheel, so the only thing at runtime that
    knows which of the two wheels this is, is its own installed distribution
    (ticket 23)."""
    assert (
        _bootstrap.shipping_distribution(["numpy", "dolfinx-solver-real"])
        == "dolfinx-solver-real"
    )
    assert _bootstrap.scalar_variant("dolfinx-solver-real") == "real"


def test_the_meta_package_is_not_a_variant():
    """The bare `dolfinx-solver` depends on a variant, it is not one."""
    assert _bootstrap.shipping_distribution(["dolfinx-solver"]) is None


def test_a_payload_that_cannot_tell_which_wheel_it_is_says_nothing():
    """A source checkout has no variant distribution; two of them installed at
    once cannot be told apart either."""
    assert _bootstrap.shipping_distribution(["numpy"]) is None
    assert (
        _bootstrap.shipping_distribution(
            ["dolfinx-solver-complex", "dolfinx-solver-real"]
        )
        is None
    )


def test_the_conflict_message_names_the_distribution_to_reinstall():
    problem = _bootstrap.conflicting_distribution_problem(
        ["petsc4py", "dolfinx-solver-real"]
    )

    assert problem is not None
    assert "reinstall dolfinx-solver-real" in problem
    assert "real-scalar" in problem
    assert "complex" not in problem


def test_the_conflict_message_claims_no_variant_it_cannot_know():
    """Naming the wrong one is worse than naming none: `reinstall
    dolfinx-solver-complex` would install a complex stack over a real one."""
    problem = _bootstrap.conflicting_distribution_problem(["petsc4py"])

    assert problem is not None
    assert "complex" not in problem
    assert "real-scalar" not in problem


def test_the_foreign_petsc4py_message_names_the_variant_when_it_knows_it():
    origin = Path("/usr/lib/python3/dist-packages/petsc4py/__init__.py")

    problem = _bootstrap.foreign_petsc4py_problem(
        origin, PACKAGE_DIR, distribution="dolfinx-solver-real"
    )

    assert problem is not None
    assert "real-scalar" in problem
    assert "complex" not in problem


def test_the_foreign_petsc4py_message_claims_no_variant_it_cannot_know():
    origin = Path("/usr/lib/python3/dist-packages/petsc4py/__init__.py")

    problem = _bootstrap.foreign_petsc4py_problem(origin, PACKAGE_DIR)

    assert problem is not None
    assert "complex" not in problem


def test_the_installed_distribution_is_read_once_for_both_messages():
    """Whichever guard fires, it is the same lookup behind it (ticket 23)."""
    calls: list[int] = []

    def installed_distribution_names() -> list[str]:
        calls.append(1)
        return ["dolfinx-solver-real", "petsc4py"]

    with pytest.raises(ImportError) as caught:
        _bootstrap.bootstrap(
            import_module=_importer({}),
            petsc4py_origin=lambda: None,
            package_dir=PACKAGE_DIR,
            installed_distribution_names=installed_distribution_names,
        )

    assert "dolfinx-solver-real" in str(caught.value)
    assert len(calls) == 1


def test_the_way_out_is_never_the_meta_package():
    """`dolfinx-solver` resolves to the complex wheel, so offering it to a
    real install is the same wrong advice as naming the variant wrongly."""
    origin = Path("/usr/lib/python3/dist-packages/petsc4py/__init__.py")

    problem = _bootstrap.foreign_petsc4py_problem(
        origin, PACKAGE_DIR, distribution="dolfinx-solver-real"
    )

    assert problem is not None
    assert "install dolfinx-solver-real in an environment of its own" in problem


def test_the_way_out_names_no_distribution_when_there_is_none_to_name():
    origin = Path("/usr/lib/python3/dist-packages/petsc4py/__init__.py")

    problem = _bootstrap.foreign_petsc4py_problem(origin, PACKAGE_DIR)

    assert problem is not None
    assert "install this wheel in an environment of its own" in problem
