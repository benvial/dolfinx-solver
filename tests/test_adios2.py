"""The ADIOS2 build: nothing left to what the build image happens to carry."""

from pathlib import Path

import pytest

from wheelbuild import adios2

GOOD_BUILD = adios2.Build(
    version=adios2.ADIOS2_VERSION,
    defines=frozenset(
        {adios2.feature_define(feature) for feature in adios2.ENABLED_FEATURES}
        | {"ADIOS2_HAVE_BP5"}
    ),
    soname=adios2.soname(),
    needed=frozenset({"libmpi.so.12", "libhdf5.so.310", "libc.so.6"}),
)


def _install(prefix: Path, relatives=adios2.REQUIRED_ARTEFACTS) -> Path:
    """Lay out an ADIOS2 install prefix."""
    for relative in relatives:
        path = prefix / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return prefix


def test_the_source_url_names_the_pinned_release():
    assert adios2.source_url().endswith(f"v{adios2.ADIOS2_VERSION}.tar.gz")


def test_the_tarball_unpacks_to_a_directory_the_url_does_not_name():
    """The archive is ``v2.12.1.tar.gz`` and the tree inside is not."""
    assert adios2.source_dir_name("2.12.1") == "ADIOS2-2.12.1"


def test_the_soname_is_versioned_by_series():
    assert adios2.soname("2.12.1") == "libadios2_cxx_mpi.so.2.12"


def test_the_configure_option_and_the_header_macro_spell_features_differently():
    """``ADIOS2_USE_ZeroMQ`` becomes ``ADIOS2_HAVE_ZEROMQ``."""
    assert adios2.feature_define("ZeroMQ") == "ADIOS2_HAVE_ZEROMQ"
    assert adios2.feature_define("MPI") == "ADIOS2_HAVE_MPI"


def test_configure_names_every_optional_feature_one_way_or_the_other():
    """A feature left at AUTO is one the build image decides (spec §8)."""
    arguments = adios2.configure_arguments(
        source_dir=Path("/build/ADIOS2-2.12.1"),
        build_dir=Path("/build/ADIOS2-2.12.1-build"),
        prefix=Path("/build/install"),
        mpi_prefix=Path("/build/install"),
    )

    for feature in adios2.ENABLED_FEATURES:
        assert f"-DADIOS2_USE_{feature}=ON" in arguments
    for feature in adios2.DISABLED_FEATURES:
        assert f"-DADIOS2_USE_{feature}=OFF" in arguments


def test_no_feature_is_both_enabled_and_disabled():
    assert not set(adios2.ENABLED_FEATURES) & set(adios2.DISABLED_FEATURES)


def test_configure_builds_against_the_prefixs_mpi_and_parallel_hdf5():
    arguments = adios2.configure_arguments(
        source_dir=Path("/build/ADIOS2-2.12.1"),
        build_dir=Path("/build/ADIOS2-2.12.1-build"),
        prefix=Path("/build/install"),
        mpi_prefix=Path("/build/install"),
    )

    assert "-DCMAKE_C_COMPILER=/build/install/bin/mpicc" in arguments
    assert "-DCMAKE_CXX_COMPILER=/build/install/bin/mpicxx" in arguments
    assert "-DHDF5_ROOT=/build/install" in arguments
    assert "-DHDF5_PREFER_PARALLEL=ON" in arguments


def test_configure_installs_into_lib_rather_than_lib64():
    arguments = adios2.configure_arguments(
        source_dir=Path("/build/ADIOS2-2.12.1"),
        build_dir=Path("/build/ADIOS2-2.12.1-build"),
        prefix=Path("/build/install"),
        mpi_prefix=Path("/build/install"),
    )

    assert "-DCMAKE_INSTALL_PREFIX=/build/install" in arguments
    assert "-DCMAKE_INSTALL_LIBDIR=lib" in arguments


def test_the_release_is_read_out_of_the_installed_header():
    header = (
        '#define ADIOS2_VERSION_STR "2.12.1"\n'
        "#define ADIOS2_VERSION_MAJOR 2\n"
        "#define ADIOS2_VERSION_MINOR 12\n"
        "#define ADIOS2_VERSION_PATCH 1\n"
    )

    assert adios2.version_in_header(header) == "2.12.1"


def test_a_header_without_a_full_version_reports_none():
    assert adios2.version_in_header("#define ADIOS2_VERSION_MAJOR 2\n") is None


def test_a_build_that_matches_on_every_count_passes():
    assert adios2.build_problem(GOOD_BUILD) is None


def test_a_stale_adios2_left_in_a_warm_cache_is_reported():
    problem = adios2.build_problem(GOOD_BUILD._replace(version="2.10.2"))

    assert problem is not None
    assert "2.10.2" in problem


def test_an_adios2_without_mpi_is_reported():
    """DOLFINx asks for the CXX and MPI components together."""
    problem = adios2.build_problem(
        GOOD_BUILD._replace(defines=frozenset({"ADIOS2_HAVE_HDF5"}))
    )

    assert problem is not None
    assert "MPI" in problem


def test_an_adios2_that_found_a_compressor_on_the_build_image_is_reported():
    problem = adios2.build_problem(
        GOOD_BUILD._replace(defines=GOOD_BUILD.defines | {"ADIOS2_HAVE_BLOSC2"})
    )

    assert problem is not None
    assert "Blosc2" in problem


def test_a_library_with_an_unexpected_soname_is_reported():
    problem = adios2.build_problem(
        GOOD_BUILD._replace(soname="libadios2_cxx_mpi.so.2.10")
    )

    assert problem is not None
    assert adios2.soname() in problem


def test_validate_returns_the_prefix_when_everything_holds(tmp_path):
    prefix = _install(tmp_path)

    assert adios2.validate(prefix, GOOD_BUILD) is prefix


def test_validate_refuses_an_install_without_the_mpi_libraries(tmp_path):
    """A prefix carrying only the serial libraries is a failed MPI search."""
    relatives = [
        path
        for path in adios2.REQUIRED_ARTEFACTS
        if path != Path(f"lib/{adios2.CXX_LIBRARY}.so")
    ]
    prefix = _install(tmp_path, relatives)

    with pytest.raises(FileNotFoundError, match="libadios2_cxx_mpi"):
        adios2.validate(prefix, GOOD_BUILD)


def test_validate_refuses_a_stale_adios2(tmp_path):
    prefix = _install(tmp_path)

    with pytest.raises(ValueError, match=r"2\.10\.2"):
        adios2.validate(prefix, GOOD_BUILD._replace(version="2.10.2"))
