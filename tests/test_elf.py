"""Reading what a shared library declares, out of `nm` and `readelf` output.

The MPICH interop check (ADR-0001) is a symbol-level comparison between the
`libmpifort` this build vendors and the `libmpi` the PyPI wheel supplies, so
the parsing that feeds it has to be exact about which symbols a library needs
and which it provides — and about which library it asks the loader for.
"""

from wheelbuild import elf

# A representative slice of `nm -D` output: a defined function, a defined
# object, an undefined reference, a versioned libc reference, and the weak
# undefined symbol every ELF object carries.
NM_OUTPUT = """\
0000000000037230 T MPI_Init
0000000000037240 T mpi_init_
0000000000052110 B MPIR_Process
                 U MPIR_Comm_free_impl
                 U memcpy@GLIBC_2.14
                 w __gmon_start__
"""


def test_defined_symbols_are_the_ones_the_library_provides():
    assert elf.defined_symbols(NM_OUTPUT) == {
        "MPI_Init",
        "mpi_init_",
        "MPIR_Process",
    }


def test_undefined_symbols_are_the_ones_the_library_needs():
    assert elf.undefined_symbols(NM_OUTPUT) == {
        "MPIR_Comm_free_impl",
        "memcpy",
        "__gmon_start__",
    }


def test_a_version_suffix_is_not_part_of_the_symbol_name():
    """`memcpy@GLIBC_2.14` and `memcpy` are one symbol with one name."""
    assert "memcpy" in elf.undefined_symbols(NM_OUTPUT)
    assert not any("@" in name for name in elf.undefined_symbols(NM_OUTPUT))


def test_weak_undefined_symbols_count_as_undefined():
    """Lowercase `w` is an undefined weak symbol; uppercase `W` is defined."""
    weak_undefined = "                 w __gmon_start__\n"
    weak_defined = "0000000000000abc W __gmon_start__\n"

    assert "__gmon_start__" in elf.undefined_symbols(weak_undefined)
    assert "__gmon_start__" in elf.defined_symbols(weak_defined)


def test_blank_and_header_lines_are_ignored():
    noise = "\nlibmpifort.so.12:\n" + NM_OUTPUT
    assert elf.defined_symbols(noise) == elf.defined_symbols(NM_OUTPUT)


def test_only_mpi_symbols_are_expected_to_come_from_libmpi():
    """libc and the loader supply the rest, so they are not libmpi's job."""
    needed = elf.undefined_symbols(NM_OUTPUT)

    assert elf.mpi_symbols(needed) == {"MPIR_Comm_free_impl"}


def test_every_mpich_internal_prefix_counts_as_an_mpi_symbol():
    names = {
        "MPI_Init",
        "PMPI_Init",
        "MPIR_Err_return_comm",
        "MPII_Keyval_free",
        "MPIX_Query_cuda_support",
        "MPL_strdup",
        "gfortran_runtime_error",
    }

    assert elf.mpi_symbols(names) == names - {"gfortran_runtime_error"}


# `readelf -d` over the vendored libmpifort, trimmed to the entries that
# decide whether the PyPI wheel's libmpi can satisfy it.
READELF_OUTPUT = """\
Dynamic section at offset 0x2d1c8 contains 30 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libmpi.so.12]
 0x0000000000000001 (NEEDED)             Shared library: [libgfortran.so.5]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000000e (SONAME)             Library soname: [libmpifort.so.12]
 0x000000000000000c (INIT)               0x9000
"""


def test_the_soname_is_the_name_the_loader_binds_against():
    assert elf.parse_soname(READELF_OUTPUT) == "libmpifort.so.12"


def test_a_library_without_a_soname_reports_none():
    assert (
        elf.parse_soname("Dynamic section at offset 0x1 contains 0 entries:\n") is None
    )


def test_the_needed_entries_are_what_the_library_asks_the_loader_for():
    assert elf.parse_needed(READELF_OUTPUT) == {
        "libmpi.so.12",
        "libgfortran.so.5",
        "libc.so.6",
    }


def test_entries_that_are_not_names_are_not_mistaken_for_them():
    """INIT and friends carry addresses, not bracketed library names."""
    assert "0x9000" not in elf.parse_needed(READELF_OUTPUT)
