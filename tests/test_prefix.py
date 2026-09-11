"""One library directory in the shared install prefix.

The manylinux_2_34 image is AlmaLinux 9, where CMake's `GNUInstallDirs`
installs into `lib64` — ADIOS2, KaHIP and DOLFINx do — while PETSc's
configure and MPICH's build use `lib`. Both spellings have to reach the same
files or the second half of the stack cannot link the first.
"""

from pathlib import Path

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
