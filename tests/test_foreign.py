"""The environments the installed wheel has to refuse, and the one it must not."""

import importlib.metadata
import sys

from dolfinx_solver import _bootstrap
from wheeltest import foreign


def test_the_shadowing_case_builds_something_python_would_import(tmp_path):
    root = foreign.write_shadowing_petsc4py(tmp_path / "case")

    assert (root / "petsc4py" / "__init__.py").exists()


def test_the_shadowing_case_is_what_the_guard_refuses(tmp_path):
    """The fabricated environment has to trip the real judgement, not a mock."""
    root = foreign.write_shadowing_petsc4py(tmp_path / "case")

    problem = _bootstrap.foreign_petsc4py_problem(
        root / "petsc4py" / "__init__.py",
        package_dir=tmp_path / "site" / "dolfinx_solver",
    )

    assert problem is not None
    assert foreign.SHADOWING.expected in problem


def test_the_installed_case_is_metadata_the_metadata_layer_finds(tmp_path):
    root = foreign.write_installed_petsc4py(tmp_path / "case")
    sys.path.insert(0, str(root))
    try:
        names = [
            _bootstrap.canonical_name(name)
            for distribution in importlib.metadata.distributions(path=[str(root)])
            if (name := distribution.metadata["Name"]) is not None
        ]
    finally:
        sys.path.remove(str(root))

    problem = _bootstrap.conflicting_distribution_problem(names)

    assert problem is not None
    assert foreign.INSTALLED.expected in problem


def test_a_refused_environment_that_imported_anyway_fails():
    problem = foreign.outcome_problem(foreign.SHADOWING, 0, "")

    assert problem is not None
    assert "has to fail" in problem


def test_a_refused_environment_that_failed_for_another_reason_fails():
    """A traceback from somewhere else is not the guard doing its job."""
    problem = foreign.outcome_problem(
        foreign.SHADOWING,
        1,
        "ImportError: libpetsc.so.3.25: cannot open shared object file",
    )

    assert problem is not None
    assert "not with the message the guard raises" in problem


def test_a_refused_environment_that_was_refused_passes():
    assert (
        foreign.outcome_problem(
            foreign.INSTALLED,
            1,
            f"ImportError: ... {foreign.INSTALLED.expected} petsc4py",
        )
        is None
    )


def test_the_clean_environment_being_refused_fails():
    """A guard that fires on a correct install fires for every user."""
    problem = foreign.outcome_problem(foreign.CLEAN, 1, "ImportError: no mpi4py")

    assert problem is not None
    assert "every user installs into" in problem


def test_the_clean_environment_importing_passes():
    assert foreign.outcome_problem(foreign.CLEAN, 0, "") is None


def test_the_clean_case_inherits_no_module_path():
    """It has to be the environment a user gets, not the suite's."""
    environment = foreign.child_environment(None)

    assert "PYTHONPATH" not in environment


def test_a_fabricated_case_goes_ahead_of_the_installed_packages(tmp_path):
    environment = foreign.child_environment(tmp_path)

    assert environment["PYTHONPATH"] == str(tmp_path)


def test_every_case_is_built_or_is_the_clean_one():
    for case in foreign.CASES:
        assert case.refused == (case.name in foreign.BUILDERS)


def test_the_child_does_not_import_the_repositorys_own_copy():
    """The suite is run from a checkout holding the source of this package."""
    assert "-P" in foreign.SAFE_PATH
