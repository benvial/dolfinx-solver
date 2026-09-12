"""The in-process proof: one MPI, complex scalars, the staged bindings."""

from types import SimpleNamespace

from wheelbuild import import_check, mpich

# Four mappings of one file, as the loader leaves them, plus an unrelated
# library and the anonymous mappings that make up most of the file.
MAPS = """\
7f0000000000-7f0000021000 r--p 00000000 fd:01 100 /venv/lib/libmpi.so.12
7f0000021000-7f0000200000 r-xp 00021000 fd:01 100 /venv/lib/libmpi.so.12
7f0000200000-7f0000300000 r--p 00200000 fd:01 100 /venv/lib/libmpi.so.12
7f0000300000-7f0000310000 rw-p 00300000 fd:01 100 /venv/lib/libmpi.so.12
7f0000400000-7f0000500000 r-xp 00000000 fd:01 101 /install/lib/libpetsc.so.3.25
7f0000600000-7f0000700000 rw-p 00000000 00:00 0
7f0000800000-7f0000900000 rw-p 00000000 00:00 0                          [heap]
"""

#: The same process with the vendored MPICH loaded beside the wheel's — what
#: the whole import-order mechanism exists to prevent (spec §5).
TWO_MPI_MAPS = MAPS + (
    "7f0001000000-7f0001100000 r-xp 00000000 fd:01 102 /install/lib/libmpi.so.12.6.1\n"
)


def test_one_libmpi_is_found_however_many_times_it_is_mapped():
    assert import_check.mapped_libraries(MAPS, mpich.MPI_SONAME) == {
        "/venv/lib/libmpi.so.12"
    }


def test_a_versioned_filename_is_the_same_library_as_its_soname():
    """The prefix's copy is `libmpi.so.12.6.1`; the wheel's is `libmpi.so.12`."""
    assert import_check.mapped_libraries(TWO_MPI_MAPS, mpich.MPI_SONAME) == {
        "/venv/lib/libmpi.so.12",
        "/install/lib/libmpi.so.12.6.1",
    }


def test_a_library_that_merely_starts_with_the_soname_is_not_it():
    maps = "7f00-7f01 r-xp 0 fd:01 1 /venv/lib/libmpi.so.120\n"

    assert import_check.mapped_libraries(maps, mpich.MPI_SONAME) == set()


def test_two_mpi_runtimes_in_one_process_are_reported():
    problem = import_check.mpi_problem(TWO_MPI_MAPS)

    assert problem is not None
    assert "/install/lib/libmpi.so.12.6.1" in problem


def test_one_mpi_runtime_is_what_the_wheel_needs():
    assert import_check.mpi_problem(MAPS) is None


def test_an_unmapped_libmpi_is_reported():
    """mpi4py imported first is what maps it; nothing else in the stack does."""
    problem = import_check.mpi_problem("")

    assert problem is not None
    assert mpich.MPI_SONAME in problem


def test_a_real_scalar_petsc_is_reported():
    """`float` stands in for `numpy.float64`, which refuses a complex too."""
    problem = import_check.scalar_problem(float)

    assert problem is not None
    assert "complex" in problem


def test_a_complex_scalar_petsc_is_what_this_distribution_ships():
    assert import_check.scalar_problem(complex) is None


def test_a_binding_imported_from_outside_the_staged_tree_is_reported(tmp_path):
    """A petsc4py from site-packages would prove nothing about the wheel's."""
    elsewhere = "/venv/lib/python3.12/site-packages/petsc4py/__init__.py"
    module = SimpleNamespace(__file__=elsewhere)

    problem = import_check.origin_problem("petsc4py", module, tmp_path)

    assert problem is not None
    assert str(tmp_path) in problem


def test_a_binding_imported_out_of_the_staged_tree_passes(tmp_path):
    module = SimpleNamespace(__file__=str(tmp_path / "petsc4py" / "__init__.py"))

    assert import_check.origin_problem("petsc4py", module, tmp_path) is None


def test_the_bindings_are_imported_after_mpi(tmp_path):
    """The order is the mechanism: mpi4py's libmpi is the one everything binds."""
    imported = []

    def import_module(name):
        imported.append(name)
        if name == "mpi4py.MPI":
            return SimpleNamespace()
        return SimpleNamespace(
            __file__=str(tmp_path / name.partition(".")[0] / "__init__.py"),
            ScalarType=complex,
        )

    import_check.check(site=tmp_path, import_module=import_module, maps_text=MAPS)

    assert imported[0] == "mpi4py.MPI"
    assert imported[1:] == ["petsc4py.PETSc", "slepc4py.SLEPc"]


def test_the_check_passes_on_a_stack_that_holds_together(tmp_path):
    def import_module(name):
        return SimpleNamespace(
            __file__=str(tmp_path / name.partition(".")[0] / "__init__.py"),
            ScalarType=complex,
        )

    assert (
        import_check.check(site=tmp_path, import_module=import_module, maps_text=MAPS)
        is None
    )


def test_a_missing_binding_is_reported_rather_than_raised(tmp_path):
    def import_module(name):
        if name == "slepc4py.SLEPc":
            raise ImportError("libslepc.so.3.25: cannot open shared object file")
        return SimpleNamespace(
            __file__=str(tmp_path / name.partition(".")[0] / "__init__.py"),
            ScalarType=complex,
        )

    problem = import_check.check(
        site=tmp_path, import_module=import_module, maps_text=MAPS
    )

    assert problem is not None
    assert "libslepc.so.3.25" in problem


def test_an_mpi4py_that_will_not_import_is_reported(tmp_path):
    """Without it nothing later in the check would mean anything."""

    def import_module(name):
        raise ImportError(f"no module named {name}")

    problem = import_check.check(
        site=tmp_path, import_module=import_module, maps_text=MAPS
    )

    assert problem is not None
    assert import_check.MPI_MODULE in problem
