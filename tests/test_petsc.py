"""The complex PETSc build: what it configures, and what it must prove.

PETSc's configure line is the whole dependency stack (spec §8), so most of
what is checked here is that the line still says what the licensing audit and
the ABI decisions require — and that a build which quietly came out different
is caught by reading the prefix rather than trusted.
"""

from pathlib import Path

import pytest

from wheelbuild import petsc

# What a correct complex build reports about itself.
GOOD_BUILD = petsc.Build(
    version=petsc.PETSC_VERSION,
    defines=frozenset(
        {
            *petsc.REQUIRED_DEFINES,
            petsc.COMPLEX_DEFINE,
            petsc.PRECISION_DEFINE,
            "PETSC_HAVE_MPI",
        }
    ),
    soname=petsc.soname(),
    needed=frozenset({"libmpi.so.12", "libm.so.6", "libc.so.6"}),
)


def _install(prefix: Path, relatives=petsc.REQUIRED_ARTEFACTS) -> Path:
    """Lay out a PETSc install prefix with the given files in it."""
    for relative in relatives:
        path = prefix / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return prefix


def _arguments(**overrides):
    defaults = {
        "source_dir": Path("/build/petsc-3.25.5"),
        "prefix": Path("/build/install"),
        "mpi_prefix": Path("/build/install"),
        "jobs": 8,
    }
    return petsc.configure_arguments(**{**defaults, **overrides})


def test_the_source_url_names_the_pinned_release():
    assert petsc.source_url().endswith(f"petsc-{petsc.PETSC_VERSION}.tar.gz")


def test_the_series_is_major_and_minor():
    """petsc4py pins PETSc at minor granularity, so that is the coupling."""
    assert petsc.series("3.25.5") == "3.25"
    assert petsc.series("3.24.6") == "3.24"


def test_the_soname_is_versioned_by_series_not_by_release():
    """A patch bump must not orphan everything already linked against it."""
    assert petsc.soname("3.25.5") == "libpetsc.so.3.25"
    assert petsc.soname("3.25.0") == petsc.soname("3.25.5")


def test_the_build_arch_names_the_scalar_type():
    """A cache may hold both variants' build trees."""
    assert petsc.build_arch("complex") == "arch-complex"
    assert petsc.build_arch("real") != petsc.build_arch("complex")


def test_configure_builds_the_whole_downloaded_stack():
    arguments = _arguments()

    for package in petsc.DOWNLOADED_PACKAGES:
        assert f"--download-{package}=1" in arguments


def test_configure_never_asks_for_parmetis():
    """Non-free: a wheel carrying it could not be published (spec §7)."""
    arguments = _arguments()

    assert "--with-parmetis=0" in arguments
    assert "parmetis" not in petsc.DOWNLOADED_PACKAGES


def test_configure_bakes_in_complex_scalars_by_default():
    assert "--with-scalar-type=complex" in _arguments()


def test_the_real_variant_is_the_same_line_with_the_scalar_flipped():
    """Spec §6: scalar type is a build parameter, not a runtime switch."""
    complex_arguments = _arguments()
    real_arguments = _arguments(scalar_type="real")

    differences = set(complex_arguments) ^ set(real_arguments)

    assert differences == {
        "--with-scalar-type=complex",
        "--with-scalar-type=real",
        "PETSC_ARCH=arch-complex",
        "PETSC_ARCH=arch-real",
    }


def test_configure_pins_the_rest_of_the_abi_explicitly():
    """A PETSc default is someone else's decision to change between releases."""
    arguments = _arguments()

    assert "--with-precision=double" in arguments
    assert "--with-64-bit-indices=0" in arguments
    assert "--with-64-bit-blas-indices=0" in arguments
    assert "--with-debugging=0" in arguments
    assert "--with-shared-libraries=1" in arguments


def test_configure_builds_openblas_for_an_unknown_machine():
    """Otherwise the wheel dies with an illegal instruction on an older CPU."""
    arguments = _arguments()

    (openblas,) = [
        argument
        for argument in arguments
        if argument.startswith("--download-openblas-make-options=")
    ]

    assert "DYNAMIC_ARCH=1" in openblas


def test_configure_tunes_nothing_to_the_build_machine():
    """The wheel is manylinux; -march would pin it to the builder's CPU."""
    assert not any("march" in argument for argument in _arguments())


def test_configure_leaves_all_parallelism_to_mpi():
    """A threaded BLAS under an MPI rank oversubscribes by the thread count."""
    arguments = _arguments()

    assert "--with-openmp=0" in arguments
    assert "--download-openblas-use-pthreads=0" in arguments


def test_configure_uses_the_mpi_wrappers_from_the_shared_prefix():
    """Their mpif.h is what ticket 02 built the Fortran half of MPICH for."""
    arguments = _arguments(mpi_prefix=Path("/build/install"))

    assert "--with-cc=/build/install/bin/mpicc" in arguments
    assert "--with-cxx=/build/install/bin/mpicxx" in arguments
    assert "--with-fc=/build/install/bin/mpifort" in arguments
    assert "--with-mpi=1" in arguments


def test_configure_keeps_a_fortran_compiler_without_fortran_bindings():
    """MUMPS and ScaLAPACK are Fortran; nothing calls PETSc from it."""
    arguments = _arguments()

    assert "--with-fortran-bindings=0" in arguments
    assert any(argument.startswith("--with-fc=") for argument in arguments)


def test_configure_installs_into_the_prefix_it_is_given():
    arguments = _arguments()

    assert arguments[0] == "/build/petsc-3.25.5/configure"
    assert "--prefix=/build/install" in arguments


def test_the_check_target_is_aimed_at_the_prefix_install():
    """PETSC_ARCH must be empty, or make looks inside an arch that is gone."""
    assert petsc.check_arguments(Path("/build/install")) == [
        "make",
        "PETSC_DIR=/build/install",
        "PETSC_ARCH=",
        "check",
    ]


def test_a_complete_install_has_nothing_missing(tmp_path):
    assert petsc.missing_artefacts(_install(tmp_path)) == []


def test_an_install_that_never_reached_make_install_is_reported(tmp_path):
    variables = Path("lib/petsc/conf/petscvariables")
    relatives = [path for path in petsc.REQUIRED_ARTEFACTS if path != variables]

    assert variables in petsc.missing_artefacts(_install(tmp_path, relatives))


def test_the_release_is_read_out_of_the_installed_header():
    header = (
        "#define PETSC_VERSION_RELEASE 1\n"
        "#define PETSC_VERSION_MAJOR 3\n"
        "#define PETSC_VERSION_MINOR 25\n"
        "#define PETSC_VERSION_SUBMINOR 5\n"
    )

    assert petsc.version_in_header(header) == "3.25.5"


def test_a_header_without_a_full_version_reports_none():
    assert petsc.version_in_header("#define PETSC_VERSION_MAJOR 3\n") is None


def test_a_build_that_matches_on_every_count_passes():
    assert petsc.build_problem(GOOD_BUILD) is None


def test_a_build_of_the_wrong_release_is_reported():
    problem = petsc.build_problem(GOOD_BUILD._replace(version="3.24.6"))

    assert problem is not None
    assert "3.24.6" in problem


def test_a_libpetsc_with_an_unexpected_soname_is_reported():
    """Two PETSc copies in one process are two PETSc global states."""
    problem = petsc.build_problem(GOOD_BUILD._replace(soname="libpetsc.so.3.24"))

    assert problem is not None
    assert petsc.soname() in problem


def test_a_real_build_in_the_complex_distribution_is_reported():
    real = GOOD_BUILD._replace(defines=GOOD_BUILD.defines - {petsc.COMPLEX_DEFINE})

    problem = petsc.build_problem(real)

    assert problem is not None
    assert "real" in problem


def test_a_complex_build_in_the_real_distribution_is_reported():
    problem = petsc.build_problem(GOOD_BUILD, scalar_type="real")

    assert problem is not None
    assert "complex" in problem


def test_the_real_variant_accepts_a_real_build():
    real = GOOD_BUILD._replace(defines=GOOD_BUILD.defines - {petsc.COMPLEX_DEFINE})

    assert petsc.build_problem(real, scalar_type="real") is None


@pytest.mark.parametrize("define", sorted(petsc.REQUIRED_DEFINES))
def test_a_solver_that_quietly_failed_to_configure_is_reported(define):
    """Configure logs a package it could not build and carries on."""
    build = GOOD_BUILD._replace(defines=GOOD_BUILD.defines - {define})

    problem = petsc.build_problem(build)

    assert problem is not None
    assert petsc.REQUIRED_DEFINES[define] in problem


@pytest.mark.parametrize("define", sorted(petsc.FORBIDDEN_DEFINES))
def test_a_define_that_would_sink_the_wheel_is_reported(define):
    build = GOOD_BUILD._replace(defines=GOOD_BUILD.defines | {define})

    problem = petsc.build_problem(build)

    assert problem is not None
    assert define in problem


def test_a_build_without_double_precision_is_reported():
    build = GOOD_BUILD._replace(defines=GOOD_BUILD.defines - {petsc.PRECISION_DEFINE})

    problem = petsc.build_problem(build)

    assert problem is not None
    assert petsc.PRECISION_DEFINE in problem


def test_a_prefix_without_parmetis_has_none_to_report(tmp_path):
    assert petsc.parmetis_artefacts(_install(tmp_path)) == []


def test_ptscotchs_parmetis_interface_is_not_parmetis(tmp_path):
    """It is SCOTCH's own CeCILL-C emulation of the API, and we do ship it."""
    prefix = _install(tmp_path)
    (prefix / "lib" / "libptscotchparmetisv3.a").touch()

    assert petsc.parmetis_artefacts(prefix) == []


def test_parmetis_in_the_tree_is_reported(tmp_path):
    prefix = _install(tmp_path)
    (prefix / "lib" / "libparmetis.so").touch()

    assert petsc.parmetis_artefacts(prefix) == [Path("lib/libparmetis.so")]


def test_validate_returns_the_prefix_when_everything_holds(tmp_path):
    prefix = _install(tmp_path)

    assert petsc.validate(prefix, GOOD_BUILD) is prefix


def test_validate_refuses_an_incomplete_install(tmp_path):
    relatives = [
        path for path in petsc.REQUIRED_ARTEFACTS if path != Path("lib/libpetsc.so")
    ]
    prefix = _install(tmp_path, relatives)

    with pytest.raises(FileNotFoundError, match="libpetsc"):
        petsc.validate(prefix, GOOD_BUILD)


def test_validate_refuses_a_prefix_carrying_parmetis(tmp_path):
    prefix = _install(tmp_path)
    (prefix / "include" / "parmetis.h").touch()

    with pytest.raises(ValueError, match="ParMETIS"):
        petsc.validate(prefix, GOOD_BUILD)


def test_validate_carries_the_scalar_type_it_is_given(tmp_path):
    """Otherwise the real variant is validated as though it were complex."""
    prefix = _install(tmp_path)
    real = GOOD_BUILD._replace(defines=GOOD_BUILD.defines - {petsc.COMPLEX_DEFINE})

    assert petsc.validate(prefix, real, scalar_type="real") is prefix
    with pytest.raises(ValueError, match="real-scalar build"):
        petsc.validate(prefix, real)


def test_validate_refuses_a_build_missing_a_solver(tmp_path):
    prefix = _install(tmp_path)
    build = GOOD_BUILD._replace(defines=GOOD_BUILD.defines - {"PETSC_HAVE_MUMPS"})

    with pytest.raises(ValueError, match="MUMPS"):
        petsc.validate(prefix, build)


#: The workflow's matrix picks the variant per job through this variable, and
#: `declared_scalar_type` is the one place it is read (ticket 21).
VARIABLE = petsc.SCALAR_TYPE_VARIABLE


def test_the_declared_variant_is_the_drivers_own_when_nothing_chooses_one():
    """A developer's build, and every unit test here, gets the default."""
    assert petsc.declared_scalar_type({}) == petsc.DEFAULT_SCALAR_TYPE
    assert petsc.DEFAULT_SCALAR_TYPE in petsc.SCALAR_TYPES


@pytest.mark.parametrize("variant", petsc.SCALAR_TYPES)
def test_the_environment_picks_the_variant_the_job_builds(variant):
    assert petsc.declared_scalar_type({VARIABLE: variant}) == variant


def test_an_empty_value_is_not_a_variant_and_is_not_a_failure():
    """An unset variable and one set to nothing mean the same thing, which is
    what an expression expanding to no matrix value leaves behind."""
    assert petsc.declared_scalar_type({VARIABLE: ""}) == petsc.DEFAULT_SCALAR_TYPE
    assert petsc.declared_scalar_type({VARIABLE: "  "}) == petsc.DEFAULT_SCALAR_TYPE


def test_a_variant_the_drivers_do_not_name_is_refused_rather_than_assumed():
    """Ticket 24's reading: a value nothing can build must refuse both
    variants rather than be read as "not complex" by every later check."""
    with pytest.raises(ValueError, match=VARIABLE):
        petsc.declared_scalar_type({VARIABLE: "Complex"})

    with pytest.raises(ValueError, match="double"):
        petsc.declared_scalar_type({VARIABLE: "double"})


def test_the_module_constant_is_what_the_environment_declared():
    """Every module that reads the variant reads this one, at import."""
    assert petsc.declared_scalar_type() == petsc.SCALAR_TYPE
