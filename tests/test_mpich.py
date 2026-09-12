"""The binding-half MPICH build: what it configures, and what it must prove.

The wheel vendors MPICH's two binding shims — `libmpifort` and `libmpicxx` —
and no `libmpi` of its own (ADR-0001, ADR-0003, ticket 13), so the checks here
are about the seam between the halves we build and the `libmpi` the PyPI
`mpich` wheel supplies at run time. Both shims are checked the same way,
because both fail the same way: they bind `libmpi` by a soname MPICH freezes
across series, so a mismatch is a clean link and a broken first import.
"""

from pathlib import Path

import pytest

from wheelbuild import mpich

# The facts a correct build reports about itself: the release from mpi.h, the
# soname its libmpi declares, and what each vendored shim asks the loader for
# and leaves undefined.
GOOD_BUILD = mpich.Build(
    version=mpich.MPICH_VERSION,
    mpi_soname=mpich.MPI_SONAME,
    shims=(
        mpich.Shim(
            soname=mpich.FORTRAN_SONAME,
            needed=frozenset({mpich.MPI_SONAME, "libgfortran.so.5", "libc.so.6"}),
            undefined=frozenset({"MPI_Init", "MPIR_Comm_free_impl", "memcpy"}),
        ),
        mpich.Shim(
            soname=mpich.CXX_SONAME,
            needed=frozenset({mpich.MPI_SONAME, "libstdc++.so.6", "libc.so.6"}),
            undefined=frozenset({"MPI_Finalize", "_ZdlPv"}),
        ),
    ),
)

# What `libmpi.so.12` from the PyPI mpich wheel exports, in miniature.
RUNTIME_EXPORTED = frozenset({"MPI_Init", "MPIR_Comm_free_impl", "MPI_Finalize"})


def _shim(build: mpich.Build, soname: str) -> mpich.Shim:
    """The shim of `build` that declares `soname`."""
    return next(shim for shim in build.shims if shim.soname == soname)


def _replacing(build: mpich.Build, soname: str, **changes) -> mpich.Build:
    """`build` with one of its shims changed and the other left alone."""
    return build._replace(
        shims=tuple(
            shim._replace(**changes) if shim.soname == soname else shim
            for shim in build.shims
        )
    )


def _install(prefix: Path, relatives=mpich.REQUIRED_ARTEFACTS) -> Path:
    """Lay out an MPICH install prefix with the given files in it."""
    for relative in relatives:
        path = prefix / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return prefix


def _without(relative: Path) -> list[Path]:
    """The required artefacts, minus one."""
    return [path for path in mpich.REQUIRED_ARTEFACTS if path != relative]


def test_the_source_url_names_the_pinned_release():
    assert mpich.MPICH_VERSION in mpich.source_url()
    assert mpich.source_url().endswith(f"mpich-{mpich.MPICH_VERSION}.tar.gz")


def test_the_series_is_the_major_version():
    assert mpich.series("5.0.1") == "5"
    assert mpich.series("4.2.3") == "4"


def test_configure_takes_the_pypi_wheels_own_line():
    """Differing from the wheel's configure is differing from its libmpi."""
    arguments = mpich.configure_arguments(
        source_dir=Path("/src/mpich-5.0.1"), prefix=Path("/install")
    )

    for argument in mpich.WHEEL_CONFIGURE:
        assert argument in arguments


def test_configure_asks_for_fortran_without_the_f08_bindings():
    arguments = mpich.configure_arguments(
        source_dir=Path("/src/mpich-5.0.1"), prefix=Path("/install")
    )

    assert "--disable-f08" in arguments
    assert "--disable-fortran" not in arguments
    assert "--disable-f77" not in arguments
    assert "--disable-f90" not in arguments


def test_configure_installs_into_the_prefix_it_is_given():
    arguments = mpich.configure_arguments(
        source_dir=Path("/src/mpich-5.0.1"), prefix=Path("/install")
    )

    assert arguments[0] == "/src/mpich-5.0.1/configure"
    assert "--prefix=/install" in arguments


def test_a_complete_install_has_nothing_missing(tmp_path):
    assert mpich.missing_artefacts(_install(tmp_path)) == []


def test_a_build_without_fortran_is_reported(tmp_path):
    fortran_library = Path(f"lib/{mpich.FORTRAN_SONAME}")

    missing = mpich.missing_artefacts(_install(tmp_path, _without(fortran_library)))

    assert missing == [fortran_library]


def test_a_build_without_the_fortran_headers_is_reported(tmp_path):
    """PETSc's Fortran packages compile against `mpif.h`, not just the library."""
    header = Path("include/mpif.h")

    assert header in mpich.missing_artefacts(_install(tmp_path, _without(header)))


def test_an_install_without_f08_bindings_has_none_to_report(tmp_path):
    assert mpich.f08_artefacts(_install(tmp_path)) == []


def test_f08_bindings_are_reported_wherever_they_appear(tmp_path):
    """The F08 handle types are not covered by the MPICH ABI (spec §5)."""
    _install(tmp_path)
    (tmp_path / "include" / "mpi_f08.mod").touch()

    assert mpich.f08_artefacts(tmp_path) == [Path("include/mpi_f08.mod")]


def test_the_release_is_read_out_of_the_installed_header():
    """mpi.h records what was built; a binary reports what it can load."""
    header = '#define MPICH_VERSION "5.0.1"\n#define MPICH_NUMVERSION 50001300\n'

    assert mpich.version_in_header(header) == "5.0.1"


def test_a_header_without_a_version_reports_none():
    assert mpich.version_in_header("#define MPI_VERSION 4\n") is None


def test_a_build_that_matches_the_pinned_release_passes():
    assert mpich.build_problem(GOOD_BUILD) is None


def test_a_build_from_the_wrong_series_is_reported():
    problem = mpich.build_problem(GOOD_BUILD._replace(version="4.2.3"))

    assert problem is not None
    assert "4.2.3" in problem
    assert mpich.MPICH_VERSION in problem


def test_a_build_whose_release_cannot_be_read_is_reported():
    problem = mpich.build_problem(GOOD_BUILD._replace(version=None))

    assert problem is not None
    assert "MPICH_VERSION" in problem


def test_a_libmpi_with_a_different_soname_is_reported():
    """Nothing in the wheel would ever ask the loader for that name."""
    problem = mpich.build_problem(GOOD_BUILD._replace(mpi_soname="libmpi.so.13"))

    assert problem is not None
    assert mpich.MPI_SONAME in problem


@pytest.mark.parametrize("soname", [mpich.FORTRAN_SONAME, mpich.CXX_SONAME])
def test_a_shim_that_does_not_bind_libmpi_is_reported(soname):
    """Binding by soname is how it reaches the libmpi mpi4py loaded first."""
    problem = mpich.build_problem(
        _replacing(GOOD_BUILD, soname, needed=frozenset({"libc.so.6"}))
    )

    assert problem is not None
    assert mpich.MPI_SONAME in problem
    assert soname in problem


@pytest.mark.parametrize("soname", [mpich.FORTRAN_SONAME, mpich.CXX_SONAME])
def test_a_shim_the_runtime_libmpi_satisfies_passes(soname):
    assert mpich.interop_problem(_shim(GOOD_BUILD, soname), RUNTIME_EXPORTED) is None


@pytest.mark.parametrize("soname", [mpich.FORTRAN_SONAME, mpich.CXX_SONAME])
def test_a_symbol_the_runtime_libmpi_does_not_export_is_reported(soname):
    """This is the check that actually proves the pin (ADR-0001)."""
    problem = mpich.interop_problem(
        mpich.Shim(
            soname=soname,
            needed=frozenset({mpich.MPI_SONAME}),
            undefined=frozenset({"MPI_Init", "MPIR_Removed_in_5_0"}),
        ),
        runtime_exported={"MPI_Init"},
    )

    assert problem is not None
    assert soname in problem
    assert "MPIR_Removed_in_5_0" in problem
    assert "MPI_Init" not in problem.replace("MPIR_Removed_in_5_0", "")


def test_symbols_libmpi_was_never_going_to_supply_are_not_its_problem():
    """libc and libgfortran supply these; auditwheel resolves them separately."""
    problem = mpich.interop_problem(
        mpich.Shim(
            soname=mpich.FORTRAN_SONAME,
            needed=frozenset({mpich.MPI_SONAME}),
            undefined=frozenset({"memcpy", "_gfortran_st_write", "__gmon_start__"}),
        ),
        runtime_exported=set(),
    )

    assert problem is None


def test_validate_returns_the_prefix_when_everything_holds(tmp_path):
    prefix = _install(tmp_path)

    assert mpich.validate(prefix, GOOD_BUILD, RUNTIME_EXPORTED) is prefix


def test_validate_refuses_an_install_missing_the_fortran_half(tmp_path):
    prefix = _install(tmp_path, _without(Path(f"lib/{mpich.FORTRAN_SONAME}")))

    with pytest.raises(FileNotFoundError, match="libmpifort"):
        mpich.validate(prefix, GOOD_BUILD, RUNTIME_EXPORTED)


def test_validate_refuses_f08_bindings(tmp_path):
    prefix = _install(tmp_path)
    (prefix / "include" / "mpi_f08.mod").touch()

    with pytest.raises(ValueError, match="mpi_f08"):
        mpich.validate(prefix, GOOD_BUILD, RUNTIME_EXPORTED)


def test_validate_refuses_a_wrong_series(tmp_path):
    prefix = _install(tmp_path)

    with pytest.raises(ValueError, match=r"4\.2\.3"):
        mpich.validate(prefix, GOOD_BUILD._replace(version="4.2.3"), RUNTIME_EXPORTED)


@pytest.mark.parametrize("soname", [mpich.FORTRAN_SONAME, mpich.CXX_SONAME])
def test_validate_refuses_a_shim_the_runtime_cannot_satisfy(tmp_path, soname):
    prefix = _install(tmp_path)
    build = _replacing(GOOD_BUILD, soname, undefined=frozenset({"MPIR_Gone"}))

    with pytest.raises(ValueError, match="MPIR_Gone"):
        mpich.validate(prefix, build, RUNTIME_EXPORTED)


def test_both_binding_shims_are_vendored_and_libmpi_is_not():
    """The PyPI wheel supplies libmpi; a second copy would be a second MPI.

    It supplies neither shim, which is why both ship (spec §5 as amended
    2026-09-12, ticket 13).
    """
    vendored = [str(relative) for relative in mpich.VENDORED_ARTEFACTS]

    assert any(name.endswith(mpich.FORTRAN_SONAME) for name in vendored)
    assert any(name.endswith(mpich.CXX_SONAME) for name in vendored)
    assert not any(name.endswith(mpich.MPI_SONAME) for name in vendored)


def test_a_build_without_the_cxx_shim_is_reported(tmp_path):
    """ADIOS2's C++ libraries name it, so an install without it is incomplete."""
    cxx_library = Path(f"lib/{mpich.CXX_SONAME}")

    missing = mpich.missing_artefacts(_install(tmp_path, _without(cxx_library)))

    assert missing == [cxx_library]


def test_validate_refuses_an_install_missing_the_cxx_shim(tmp_path):
    prefix = _install(tmp_path, _without(Path(f"lib/{mpich.CXX_SONAME}")))

    with pytest.raises(FileNotFoundError, match="libmpicxx"):
        mpich.validate(prefix, GOOD_BUILD, RUNTIME_EXPORTED)
