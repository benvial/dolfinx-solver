"""One library directory in the shared install prefix.

The manylinux_2_34 image is AlmaLinux 9, where CMake's `GNUInstallDirs`
installs into `lib64` — ADIOS2, KaHIP and DOLFINx do — while PETSc's
configure and MPICH's build use `lib`. Both spellings have to reach the same
files or the second half of the stack cannot link the first.
"""

from pathlib import Path

import pytest

from wheelbuild import petsc
from wheelbuild import prefix as prefix_module


def test_lib64_becomes_an_alias_of_lib(tmp_path):
    library_dir = prefix_module.unify_lib_directories(tmp_path)

    assert library_dir == tmp_path / "lib"
    assert library_dir.is_dir()
    assert (tmp_path / "lib64").is_symlink()
    assert (tmp_path / "lib64").resolve() == library_dir.resolve()


def test_anything_already_in_lib64_is_moved_into_lib(tmp_path):
    (tmp_path / "lib64").mkdir()
    (tmp_path / "lib64" / "libadios2.so").write_text("adios")

    prefix_module.unify_lib_directories(tmp_path)

    assert (tmp_path / "lib" / "libadios2.so").read_text() == "adios"
    assert (tmp_path / "lib64" / "libadios2.so").read_text() == "adios"


def test_directories_present_on_both_sides_are_merged(tmp_path):
    """Replacing one wholesale would drop the CMake configs only it has."""
    (tmp_path / "lib64" / "cmake" / "adios2").mkdir(parents=True)
    (tmp_path / "lib64" / "cmake" / "adios2" / "config.cmake").write_text("adios")
    (tmp_path / "lib" / "cmake" / "petsc").mkdir(parents=True)
    (tmp_path / "lib" / "cmake" / "petsc" / "config.cmake").write_text("petsc")

    prefix_module.unify_lib_directories(tmp_path)

    assert (tmp_path / "lib" / "cmake" / "adios2" / "config.cmake").exists()
    assert (tmp_path / "lib" / "cmake" / "petsc" / "config.cmake").exists()


def test_a_file_already_in_lib_wins(tmp_path):
    """That is the copy the build has been linking against all along."""
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "libmpifort.so.12").write_text("built")
    (tmp_path / "lib64").mkdir()
    (tmp_path / "lib64" / "libmpifort.so.12").write_text("stale")

    prefix_module.unify_lib_directories(tmp_path)

    assert (tmp_path / "lib" / "libmpifort.so.12").read_text() == "built"


def test_running_again_against_a_warm_cache_changes_nothing(tmp_path):
    """The container build re-runs this on every start, cached tree or not."""
    prefix_module.unify_lib_directories(tmp_path)
    (tmp_path / "lib" / "libmpifort.so.12").write_text("built")

    prefix_module.unify_lib_directories(tmp_path)

    assert (tmp_path / "lib" / "libmpifort.so.12").read_text() == "built"
    assert (tmp_path / "lib64").is_symlink()


def test_the_prefix_is_created_when_it_does_not_exist_yet(tmp_path):
    prefix = tmp_path / "install"

    library_dir = prefix_module.unify_lib_directories(prefix)

    assert library_dir == Path(prefix / "lib")
    assert library_dir.is_dir()


def test_a_fresh_prefix_takes_the_variant_that_claims_it(tmp_path):
    """A warm cache from before this check has no marker and is adopted."""
    prefix_module.claim_variant(tmp_path, "complex")

    assert (tmp_path / prefix_module.VARIANT_FILE).read_text().strip() == "complex"


def test_claiming_the_same_variant_again_is_what_every_warm_run_does(tmp_path):
    prefix_module.claim_variant(tmp_path, "complex")

    assert prefix_module.claim_variant(tmp_path, "complex") == tmp_path


def test_a_prefix_built_for_the_other_variant_is_refused(tmp_path):
    """Stamps are never pruned, so a re-used prefix would re-check a stale
    stack of the other scalar type forever (ticket 17)."""
    prefix_module.claim_variant(tmp_path, "complex")

    with pytest.raises(ValueError, match="complex"):
        prefix_module.claim_variant(tmp_path, "real")


def test_the_refusal_names_the_way_out(tmp_path):
    prefix_module.claim_variant(tmp_path, "real")

    with pytest.raises(ValueError, match="BUILD_ROOT"):
        prefix_module.claim_variant(tmp_path, "complex")


def test_the_variant_defaults_to_the_one_the_petsc_driver_declares(tmp_path):
    prefix_module.main(["--prefix", str(tmp_path)])

    assert (
        tmp_path / prefix_module.VARIANT_FILE
    ).read_text().strip() == petsc.SCALAR_TYPE


def test_a_warm_prefix_from_before_the_marker_is_read_from_its_stamps(tmp_path):
    """The migration case: a cached complex prefix carries no marker, and its
    PETSc stamp is what says which variant built it (ticket 17)."""
    (tmp_path / ".petsc-3.25.5-complex.installed").touch()

    with pytest.raises(ValueError, match="complex"):
        prefix_module.claim_variant(tmp_path, "real")


def test_a_warm_prefix_of_this_variant_is_adopted_and_marked(tmp_path):
    (tmp_path / ".petsc-3.25.5-complex.installed").touch()

    prefix_module.claim_variant(tmp_path, "complex")

    assert (tmp_path / prefix_module.VARIANT_FILE).read_text().strip() == "complex"


def test_a_variant_this_build_has_no_name_for_is_refused(tmp_path):
    """A typo would otherwise be written into the marker and then enforced."""
    with pytest.raises(ValueError, match="Complex"):
        prefix_module.claim_variant(tmp_path, "Complex")

    assert not (tmp_path / prefix_module.VARIANT_FILE).exists()


def test_nothing_is_laid_out_in_a_prefix_the_other_variant_owns(tmp_path):
    """The refusal comes before `lib64` is merged into `lib`."""
    prefix_module.claim_variant(tmp_path, "complex")

    assert prefix_module.main(["--prefix", str(tmp_path), "--scalar-type", "real"]) == 1
    assert not (tmp_path / "lib64").exists()
