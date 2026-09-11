"""Reading the `#define` facts an installed configuration header records."""

import pytest

from wheelbuild import macros


def test_a_define_with_a_value_keeps_it():
    assert macros.defined_macros("#define PETSC_VERSION_MINOR 25\n") == {
        "PETSC_VERSION_MINOR": "25"
    }


def test_a_bare_define_has_an_empty_value():
    assert macros.defined_macros("#define PETSC_HAVE_MUMPS\n") == {
        "PETSC_HAVE_MUMPS": ""
    }


def test_indented_and_spaced_directives_are_read():
    """Generated configuration headers are not uniformly formatted."""
    text = "  #  define PETSC_USE_COMPLEX 1\n"

    assert macros.defined_names(text) == frozenset({"PETSC_USE_COMPLEX"})


def test_a_function_like_macro_is_not_a_fact_about_the_build():
    """Its name is followed by `(`, so reading it would report the wrong name."""
    text = "#define PetscCall(x) do { } while (0)\n#define PETSC_HAVE_HDF5 1\n"

    assert macros.defined_names(text) == frozenset({"PETSC_HAVE_HDF5"})


def test_lines_that_are_not_defines_are_ignored():
    text = "#if defined(PETSC_HAVE_PARMETIS)\n#include <petscconf.h>\n#endif\n"

    assert macros.defined_macros(text) == {}


def test_a_redefinition_keeps_the_last_value_as_the_preprocessor_would():
    text = "#define SLEPC_VERSION_SUBMINOR 0\n#define SLEPC_VERSION_SUBMINOR 1\n"

    assert macros.defined_macros(text)["SLEPC_VERSION_SUBMINOR"] == "1"


def test_several_headers_are_read_as_one_set_of_facts(tmp_path):
    first = tmp_path / "petscconf.h"
    first.write_text("#define PETSC_HAVE_MUMPS 1\n", encoding="utf-8")
    second = tmp_path / "petscversion.h"
    second.write_text("#define PETSC_VERSION_MAJOR 3\n", encoding="utf-8")

    assert macros.read_defined_names(first, second) == frozenset(
        {"PETSC_HAVE_MUMPS", "PETSC_VERSION_MAJOR"}
    )


def test_a_missing_header_is_an_error_rather_than_no_features(tmp_path):
    """A prefix without its configuration header has not finished installing."""
    with pytest.raises(FileNotFoundError):
        macros.read_defined_names(tmp_path / "petscconf.h")
