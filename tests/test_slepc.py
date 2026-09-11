"""The SLEPc build: coupled to one PETSc, and checked to still be coupled."""

from pathlib import Path

import pytest

from wheelbuild import petsc, slepc

GOOD_BUILD = slepc.Build(
    version=slepc.SLEPC_VERSION,
    soname=slepc.soname(),
    needed=frozenset({petsc.soname(), "libmpi.so.12", "libc.so.6"}),
)


def _install(prefix: Path, relatives=slepc.REQUIRED_ARTEFACTS) -> Path:
    """Lay out a SLEPc install prefix, with the PETSc it binds beside it."""
    for relative in (*relatives, Path("lib") / petsc.soname()):
        path = prefix / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return prefix


def test_the_source_url_names_the_pinned_release():
    assert slepc.source_url().endswith(f"slepc-{slepc.SLEPC_VERSION}.tar.gz")


def test_the_soname_is_versioned_by_series():
    assert slepc.soname("3.25.1") == "libslepc.so.3.25"


def test_the_pinned_slepc_and_petsc_are_the_same_series():
    """The pin that fails in seconds instead of after the compile."""
    assert slepc.coupling_problem() is None


def test_a_slepc_from_another_series_than_petsc_is_reported():
    problem = slepc.coupling_problem("3.24.3", "3.25.5")

    assert problem is not None
    assert "3.24" in problem
    assert "3.25" in problem


def test_configure_installs_into_the_prefix_it_is_given():
    arguments = slepc.configure_arguments(
        source_dir=Path("/build/slepc-3.25.1"), prefix=Path("/build/install")
    )

    assert arguments[0] == "/build/slepc-3.25.1/configure"
    assert "--prefix=/build/install" in arguments


def test_configure_names_the_eigensolvers_the_wheel_does_not_ship():
    arguments = slepc.configure_arguments(
        source_dir=Path("/build/slepc-3.25.1"), prefix=Path("/build/install")
    )

    for package in slepc.DISABLED_PACKAGES:
        assert f"--with-{package}=0" in arguments


def test_the_environment_points_slepc_at_the_prefix_installed_petsc():
    """SLEPc takes its PETSc from the environment, not from a flag."""
    environment = slepc.build_environment(
        source_dir=Path("/build/slepc-3.25.1"),
        petsc_prefix=Path("/build/install"),
        environ={},
    )

    assert environment["PETSC_DIR"] == "/build/install"
    assert environment["SLEPC_DIR"] == "/build/slepc-3.25.1"


def test_an_inherited_petsc_arch_is_cleared_rather_than_left_alone():
    """A prefix-installed PETSc has no arch directory to look inside."""
    environment = slepc.build_environment(
        source_dir=Path("/build/slepc-3.25.1"),
        petsc_prefix=Path("/build/install"),
        environ={"PETSC_ARCH": "arch-somebody-elses"},
    )

    assert environment["PETSC_ARCH"] == ""


def test_the_build_directory_name_is_read_back_from_configures_own_record():
    """An empty PETSC_ARCH on the make line overrides what configure chose."""
    text = (
        "SLEPC_DIR = /build/slepc-3.25.1\n"
        "PETSC_ARCH = installed-arch-linux2-c-opt-complex\n"
    )

    assert slepc.arch_in_variables(text) == "installed-arch-linux2-c-opt-complex"


def test_variables_naming_no_arch_report_none():
    assert slepc.arch_in_variables("SLEPC_DIR = /build/slepc-3.25.1\n") is None


def test_an_unconfigured_source_tree_has_no_build_directory_to_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="slepcvariables"):
        slepc.configured_arch(tmp_path)


def test_a_configured_source_tree_names_its_build_directory(tmp_path):
    variables = tmp_path / "lib" / "slepc" / "conf" / "slepcvariables"
    variables.parent.mkdir(parents=True)
    variables.write_text("PETSC_ARCH = installed-arch\n", encoding="utf-8")

    assert slepc.configured_arch(tmp_path) == "installed-arch"


def test_the_release_is_read_out_of_the_installed_header():
    header = (
        "#define SLEPC_VERSION_MAJOR 3\n"
        "#define SLEPC_VERSION_MINOR 25\n"
        "#define SLEPC_VERSION_SUBMINOR 1\n"
    )

    assert slepc.version_in_header(header) == "3.25.1"


def test_a_header_without_a_full_version_reports_none():
    assert slepc.version_in_header("#define SLEPC_VERSION_MAJOR 3\n") is None


def test_a_build_that_matches_on_every_count_passes():
    assert slepc.build_problem(GOOD_BUILD) is None


def test_a_stale_slepc_left_in_a_warm_cache_is_reported():
    """It would link against the new libpetsc by soname without complaint."""
    problem = slepc.build_problem(GOOD_BUILD._replace(version="3.24.3"))

    assert problem is not None
    assert "3.24.3" in problem


def test_a_libslepc_with_an_unexpected_soname_is_reported():
    problem = slepc.build_problem(GOOD_BUILD._replace(soname="libslepc.so.3.24"))

    assert problem is not None
    assert slepc.soname() in problem


def test_a_libslepc_not_bound_to_our_petsc_is_reported():
    problem = slepc.build_problem(
        GOOD_BUILD._replace(needed=frozenset({"libmpi.so.12"}))
    )

    assert problem is not None
    assert petsc.soname() in problem


def test_validate_returns_the_prefix_when_everything_holds(tmp_path):
    prefix = _install(tmp_path)

    assert slepc.validate(prefix, GOOD_BUILD) is prefix


def test_validate_refuses_an_incomplete_install(tmp_path):
    relatives = [
        path for path in slepc.REQUIRED_ARTEFACTS if path != Path("lib/libslepc.so")
    ]
    prefix = _install(tmp_path, relatives)

    with pytest.raises(FileNotFoundError, match="libslepc"):
        slepc.validate(prefix, GOOD_BUILD)


def test_validate_refuses_a_prefix_whose_petsc_is_somewhere_else(tmp_path):
    """The soname alone is satisfied by any PETSc on the machine."""
    prefix = _install(tmp_path)
    (prefix / "lib" / petsc.soname()).unlink()

    with pytest.raises(FileNotFoundError, match=petsc.soname()):
        slepc.validate(prefix, GOOD_BUILD)


def test_validate_refuses_a_stale_slepc(tmp_path):
    prefix = _install(tmp_path)

    with pytest.raises(ValueError, match=r"3\.24\.3"):
        slepc.validate(prefix, GOOD_BUILD._replace(version="3.24.3"))
