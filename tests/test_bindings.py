"""The petsc4py and slepc4py builds: our own, and reachable by relative rpath."""

from pathlib import Path

import pytest

from wheelbuild import bindings, petsc, slepc

GOOD_BUILD = bindings.Build(
    runpath=(bindings.RELATIVE_RPATH,),
    needed=frozenset({petsc.soname(), "libmpi.so.12", "libc.so.6"}),
)


def _staged(site: Path, binding: bindings.Binding = bindings.PETSC4PY) -> Path:
    """Lay out a staged binding package the way an unpacked wheel leaves it."""
    for relative in bindings.required_artefacts(binding):
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return site


def test_both_bindings_come_out_of_the_tarballs_the_build_already_downloads():
    """Never the PyPI sdists: the sources ride inside PETSc's and SLEPc's."""
    assert bindings.source_dir(bindings.PETSC4PY, Path("/build/petsc-3.25.5")) == Path(
        "/build/petsc-3.25.5/src/binding/petsc4py"
    )
    assert bindings.source_dir(bindings.SLEPC4PY, Path("/build/slepc-3.25.1")) == Path(
        "/build/slepc-3.25.1/src/binding/slepc4py"
    )


def test_each_binding_names_the_vendored_library_it_has_to_reach():
    assert bindings.PETSC4PY.library_soname == petsc.soname()
    assert bindings.SLEPC4PY.library_soname == slepc.soname()


def test_the_extension_is_the_limited_api_one():
    """`PETSc.abi3.so`, not `PETSc.cpython-312-x86_64-linux-gnu.so`."""
    assert bindings.extension_path(bindings.PETSC4PY) == Path(
        "petsc4py/lib/PETSc.abi3.so"
    )
    assert bindings.extension_path(bindings.SLEPC4PY) == Path(
        "slepc4py/lib/SLEPc.abi3.so"
    )


def test_the_relative_rpath_climbs_out_of_the_extension_directory():
    """`<site>/petsc4py/lib/PETSc.abi3.so` reaching `<site>/dolfinx_solver/lib`."""
    depth = len(bindings.extension_path(bindings.PETSC4PY).parent.parts)
    climb = "$ORIGIN/" + "../" * depth + "dolfinx_solver/lib"

    assert climb == bindings.RELATIVE_RPATH


def test_the_build_environment_asks_for_a_cp312_limited_api_extension():
    environment = bindings.build_environment(
        bindings.PETSC4PY, prefix=Path("/build/install"), site=Path("/site"), environ={}
    )

    assert environment[bindings.PETSC4PY.sabi_variable] == bindings.LIMITED_API_TAG


def test_the_build_environment_points_at_the_prefix_installed_petsc():
    environment = bindings.build_environment(
        bindings.PETSC4PY, prefix=Path("/build/install"), site=Path("/site"), environ={}
    )

    assert environment["PETSC_DIR"] == "/build/install"
    assert environment["PETSC_ARCH"] == ""


def test_slepc4py_is_told_where_both_halves_of_the_stack_are():
    """Its configure reads SLEPc's installed variables and PETSc's alike."""
    environment = bindings.build_environment(
        bindings.SLEPC4PY, prefix=Path("/build/install"), site=Path("/site"), environ={}
    )

    assert environment["SLEPC_DIR"] == "/build/install"
    assert environment["PETSC_DIR"] == "/build/install"
    assert environment["PETSC_ARCH"] == ""


def test_slepc4py_builds_against_the_petsc4py_already_staged():
    environment = bindings.build_environment(
        bindings.SLEPC4PY, prefix=Path("/build/install"), site=Path("/site"), environ={}
    )

    assert "/site" in environment["PYTHONPATH"].split(":")


def test_the_build_leaves_no_bytecode_in_the_wheels_payload():
    """slepc4py's setup.py imports the staged petsc4py out of the payload."""
    environment = bindings.build_environment(
        bindings.SLEPC4PY, prefix=Path("/build/install"), site=Path("/site"), environ={}
    )

    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_the_wheel_is_built_from_the_source_tree_with_nothing_downloaded():
    arguments = bindings.wheel_arguments(
        source_dir=Path("/build/petsc-3.25.5/src/binding/petsc4py"),
        wheelhouse=Path("/build/wheelhouse"),
        python="/venv/bin/python",
    )

    assert arguments[:3] == ["/venv/bin/python", "-m", "pip"]
    assert "--no-build-isolation" in arguments
    assert "--no-deps" in arguments
    assert arguments[-1] == "/build/petsc-3.25.5/src/binding/petsc4py"


def test_a_wheel_tagged_for_one_interpreter_is_reported():
    """The wheel rides cp312-abi3; a cp312-cp312 build would not (spec §4)."""
    problem = bindings.wheel_tag_problem("petsc4py-3.25.5-cp312-cp312-linux_x86_64.whl")

    assert problem is not None
    assert "abi3" in problem


def test_an_abi3_wheel_passes_the_tag_check():
    assert (
        bindings.wheel_tag_problem("petsc4py-3.25.5-cp312-abi3-linux_x86_64.whl")
        is None
    )


def test_the_staged_tree_is_checked_for_the_extension_and_the_package():
    assert bindings.extension_path(bindings.PETSC4PY) in bindings.required_artefacts(
        bindings.PETSC4PY
    )
    assert Path("petsc4py/__init__.py") in bindings.required_artefacts(
        bindings.PETSC4PY
    )


def test_an_absolute_build_path_on_the_rpath_is_reported():
    """What upstream's setup.py bakes in, and what patchelf has to replace."""
    problem = bindings.build_problem(
        bindings.PETSC4PY, GOOD_BUILD._replace(runpath=("/build/install/lib",))
    )

    assert problem is not None
    assert "/build/install/lib" in problem


def test_an_extension_with_no_rpath_at_all_is_reported():
    problem = bindings.build_problem(bindings.PETSC4PY, GOOD_BUILD._replace(runpath=()))

    assert problem is not None
    assert bindings.RELATIVE_RPATH in problem


def test_an_extension_not_linked_against_the_vendored_library_is_reported():
    problem = bindings.build_problem(
        bindings.PETSC4PY, GOOD_BUILD._replace(needed=frozenset({"libc.so.6"}))
    )

    assert problem is not None
    assert petsc.soname() in problem


def test_slepc4py_has_to_reach_libslepc():
    problem = bindings.build_problem(bindings.SLEPC4PY, GOOD_BUILD)

    assert problem is not None
    assert slepc.soname() in problem


def test_a_binding_that_reaches_its_library_relatively_has_no_problem():
    assert bindings.build_problem(bindings.PETSC4PY, GOOD_BUILD) is None


def test_validate_returns_the_site_when_everything_holds(tmp_path):
    site = _staged(tmp_path)

    assert bindings.validate(bindings.PETSC4PY, site, GOOD_BUILD) is site


def test_validate_refuses_a_staging_tree_missing_the_extension(tmp_path):
    site = _staged(tmp_path)
    (site / bindings.extension_path(bindings.PETSC4PY)).unlink()

    with pytest.raises(FileNotFoundError, match=r"PETSc\.abi3\.so"):
        bindings.validate(bindings.PETSC4PY, site, GOOD_BUILD)


def test_validate_refuses_a_staged_upstream_distribution(tmp_path):
    """The wheel ships the import names and no distribution of those names."""
    site = _staged(tmp_path)
    (site / "petsc4py-3.25.5.dist-info").mkdir()

    with pytest.raises(ValueError, match="dist-info"):
        bindings.validate(bindings.PETSC4PY, site, GOOD_BUILD)


def test_the_staging_site_carries_the_wheels_library_directory(tmp_path):
    """The relative rpath can only be proven in the layout it is written for."""
    site = bindings.prepare_site(tmp_path)

    resolved = (site / bindings.extension_path(bindings.PETSC4PY)).parent
    for part in bindings.RELATIVE_RPATH.removeprefix("$ORIGIN/").split("/"):
        resolved = resolved / part

    assert resolved.resolve() == (tmp_path / "lib").resolve()


def test_preparing_the_site_twice_leaves_the_same_layout(tmp_path):
    """Every stage re-runs against a warm cache."""
    first = bindings.prepare_site(tmp_path)
    second = bindings.prepare_site(tmp_path)

    assert first == second
    assert (second / "dolfinx_solver" / "lib").is_symlink()


def test_the_import_check_runs_the_module_that_holds_it():
    arguments = bindings.import_check_arguments(
        site=Path("/site"), python="/venv/bin/python"
    )

    assert arguments == [
        "/venv/bin/python",
        "-m",
        "wheelbuild.import_check",
        "--site",
        "/site",
    ]


def test_the_import_check_sees_the_staged_packages_and_the_drivers(tmp_path):
    environment = bindings.import_check_environment(site=tmp_path, environ={})

    assert environment["PYTHONPATH"].split(":")[0] == str(tmp_path)
    assert str(bindings.REPO_ROOT) in environment["PYTHONPATH"].split(":")


def test_the_import_check_leaves_no_bytecode_in_the_wheels_payload():
    environment = bindings.import_check_environment(site=Path("/site"), environ={})

    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_the_import_check_runs_without_a_library_path_to_lean_on():
    """LD_LIBRARY_PATH would resolve libpetsc whatever the rpath said."""
    environment = bindings.import_check_environment(
        site=Path("/site"), environ={"LD_LIBRARY_PATH": "/build/install/lib"}
    )

    assert "LD_LIBRARY_PATH" not in environment
