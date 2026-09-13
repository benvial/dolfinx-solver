"""Finding the wheel, and the environment the suite runs it in."""

from pathlib import Path

import pytest

from wheelbuild import assemble
from wheeltest import environment


def _wheelhouse(tmp_path: Path, *names: str) -> Path:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    for name in names:
        (wheelhouse / name).write_bytes(b"PK\x03\x04")
    return wheelhouse


def test_the_publishable_wheel_is_found_among_the_build_artefacts(tmp_path):
    """The build leaves its own petsc4py and slepc4py wheels lying about."""
    wheelhouse = _wheelhouse(
        tmp_path,
        assemble.wheel_name(assemble.WHEEL_TAG),
        "petsc4py-3.25.5-cp312-abi3-linux_x86_64.whl",
        "fenics_dolfinx-0.11.0-cp312-abi3-linux_x86_64.whl",
    )

    assert environment.find_wheel(wheelhouse).name == assemble.wheel_name(
        assemble.WHEEL_TAG
    )


def test_the_wrong_wheelhouse_says_which_one_was_meant(tmp_path):
    wheelhouse = _wheelhouse(tmp_path, "petsc4py-3.25.5-cp312-abi3-linux_x86_64.whl")

    with pytest.raises(FileNotFoundError, match=r"\$build_root/wheelhouse"):
        environment.find_wheel(wheelhouse)


def test_two_of_our_wheels_are_refused_rather_than_ordered(tmp_path):
    """A stale wheel beside a fresh one is the suite proving the wrong one."""
    wheelhouse = _wheelhouse(
        tmp_path,
        assemble.wheel_name(assemble.WHEEL_TAG),
        f"{assemble.DISTRIBUTION}-0.10.0-cp312-abi3-manylinux_2_34_x86_64.whl",
    )

    with pytest.raises(ValueError, match="holds 2 wheels"):
        environment.find_wheel(wheelhouse)


def test_the_launcher_comes_from_the_wheels_mpich(tmp_path):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / environment.LAUNCHER).write_text("#!/bin/sh\n")

    found = environment.launcher(environment.venv_python(tmp_path / "venv"))

    assert found == bin_dir / environment.LAUNCHER


def test_a_venv_without_mpiexec_is_a_dependency_that_did_not_install(tmp_path):
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="mpich wheel"):
        environment.launcher(environment.venv_python(tmp_path / "venv"))


def test_the_stages_run_without_the_callers_library_search_path():
    """The rpaths are the linkage design; a search path would hide a broken one."""
    child = environment.child_environment(
        {"LD_LIBRARY_PATH": "/build/install/lib", "HOME": "/root"}
    )

    assert "LD_LIBRARY_PATH" not in child
    assert child["PYTHONPATH"] == str(environment.REPO_ROOT)
    assert child["HOME"] == "/root"


def test_the_stages_name_the_transports_instead_of_discovering_them():
    """MPICH picks its transport through UCX at MPI_Init, which for this
    payload is `import dolfinx_solver`. A host that exposes an RDMA device a
    guest may not open killed that import on one GitHub runner in six, before
    any message was sent; nothing the suite runs needs a fabric."""
    child = environment.child_environment({"HOME": "/root"})

    assert child[environment.UCX_TRANSPORTS_VARIABLE] == environment.UCX_TRANSPORTS
    assert "sm" in environment.UCX_TRANSPORTS.split(",")


def test_a_caller_who_knows_the_machine_keeps_their_own_transports():
    """A cluster whose InfiniBand works is the case this project has never
    run on, and naming TCP there would be slower for no reason."""
    child = environment.child_environment(
        {environment.UCX_TRANSPORTS_VARIABLE: "rc,ud,self"}
    )

    assert child[environment.UCX_TRANSPORTS_VARIABLE] == "rc,ud,self"


def test_an_offline_run_resolves_the_dependencies_from_the_wheelhouse(tmp_path):
    """The wheel's metadata still decides what is installed, not the index."""
    command = environment.install_arguments(
        tmp_path / "dolfinx.whl",
        ("packaging",),
        python=tmp_path / "bin" / "python",
        wheelhouse=tmp_path / "wheelhouse",
    )

    assert command[-3:] == ["--no-index", "--find-links", str(tmp_path / "wheelhouse")]
    assert "packaging" in command


def test_an_online_run_proves_the_metadata_brings_the_runtime_stack(tmp_path):
    command = environment.install_arguments(
        tmp_path / "dolfinx.whl", python=tmp_path / "bin" / "python"
    )

    assert "--no-index" not in command
    assert command[:4] == [str(tmp_path / "bin" / "python"), "-m", "pip", "install"]
