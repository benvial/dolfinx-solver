"""The KaHIP build: portable to every x86-64 machine, and MPI-linked."""

from pathlib import Path

import pytest

from wheelbuild import kahip, mpich

GOOD_BUILD = kahip.Build(
    needed=frozenset({mpich.MPI_SONAME, "libstdc++.so.6", "libc.so.6"}),
    wide_vector_registers=frozenset(),
)


def _install(prefix: Path, relatives=kahip.REQUIRED_ARTEFACTS) -> Path:
    """Lay out a KaHIP install prefix."""
    for relative in relatives:
        path = prefix / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return prefix


def test_the_source_url_names_the_pinned_release():
    assert kahip.source_url().endswith(f"v{kahip.KAHIP_VERSION}.tar.gz")


def test_the_tarball_unpacks_to_the_repositorys_own_capitalisation():
    assert kahip.source_dir_name("3.25") == "KaHIP-3.25"


def test_configure_turns_off_the_march_native_default():
    """KaHIP tunes itself to the build machine unless told not to."""
    arguments = kahip.configure_arguments(
        source_dir=Path("/build/KaHIP-3.25"),
        build_dir=Path("/build/KaHIP-3.25-build"),
        prefix=Path("/build/install"),
        mpi_prefix=Path("/build/install"),
    )

    assert "-DNONATIVEOPTIMIZATIONS=ON" in arguments


def test_configure_keeps_openmp_out_of_the_stack():
    """No threading anywhere: PETSc is configured --with-openmp=0 too."""
    arguments = kahip.configure_arguments(
        source_dir=Path("/build/KaHIP-3.25"),
        build_dir=Path("/build/KaHIP-3.25-build"),
        prefix=Path("/build/install"),
        mpi_prefix=Path("/build/install"),
    )

    assert "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON" in arguments


def test_configure_builds_the_parallel_half():
    """ParHIP is the half DOLFINx partitions a distributed mesh with."""
    arguments = kahip.configure_arguments(
        source_dir=Path("/build/KaHIP-3.25"),
        build_dir=Path("/build/KaHIP-3.25-build"),
        prefix=Path("/build/install"),
        mpi_prefix=Path("/build/install"),
    )

    assert "-DPARHIP=ON" in arguments
    assert "-DNOMPI=OFF" in arguments
    assert "-DCMAKE_CXX_COMPILER=/build/install/bin/mpicxx" in arguments


def test_a_build_that_matches_on_every_count_passes():
    assert kahip.build_problem(GOOD_BUILD) is None


def test_a_library_tuned_to_the_builders_cpu_is_reported():
    """The OpenBLAS failure mode: an illegal instruction, not a slow solve."""
    problem = kahip.build_problem(
        GOOD_BUILD._replace(wide_vector_registers=frozenset({"zmm"}))
    )

    assert problem is not None
    assert "zmm" in problem
    assert "NONATIVEOPTIMIZATIONS" in problem


def test_a_library_linked_against_openmp_is_reported():
    problem = kahip.build_problem(
        GOOD_BUILD._replace(needed=GOOD_BUILD.needed | {kahip.OPENMP_SONAME})
    )

    assert problem is not None
    assert kahip.OPENMP_SONAME in problem


def test_a_parhip_that_is_not_linked_against_mpi_is_reported():
    problem = kahip.build_problem(GOOD_BUILD._replace(needed=frozenset({"libc.so.6"})))

    assert problem is not None
    assert mpich.MPI_SONAME in problem


def test_validate_returns_the_prefix_when_everything_holds(tmp_path):
    prefix = _install(tmp_path)

    assert kahip.validate(prefix, GOOD_BUILD) is prefix


def test_validate_refuses_an_install_without_the_parallel_half(tmp_path):
    relatives = [
        path
        for path in kahip.REQUIRED_ARTEFACTS
        if path != Path("lib/libparhip_interface.so")
    ]
    prefix = _install(tmp_path, relatives)

    with pytest.raises(FileNotFoundError, match="libparhip_interface"):
        kahip.validate(prefix, GOOD_BUILD)


def test_validate_refuses_a_cpu_tuned_build(tmp_path):
    prefix = _install(tmp_path)

    with pytest.raises(ValueError, match="ymm"):
        kahip.validate(
            prefix, GOOD_BUILD._replace(wide_vector_registers=frozenset({"ymm"}))
        )
