"""The flip that makes `dolfinx-solver-real`, exercised without building it.

`wheelbuild.petsc.SCALAR_TYPE` is the single place the variant is declared
(spec §6), and everything the build decides from it has to move when it moves:
the configure line, the cache stamp names, the checks that judge a built
stack, and the prefix the stages install into. A full real build is hours, so
what is asserted here is that each of those reads the value rather than
spelling `complex` itself — which is the failure ticket 17 was raised for,
where a real build compiled for minutes and then failed a check written for
the complex one.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from wheelbuild import import_check, petsc
from wheelbuild import prefix as prefix_module

#: A complex PETSc install, as `wheelbuild.petsc.observe` would report it.
COMPLEX_BUILD = petsc.Build(
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
    needed=frozenset({"libmpi.so.12"}),
)

OTHER = {"complex": "real", "real": "complex"}
FLIPPED = OTHER[petsc.SCALAR_TYPE]


def test_the_configure_line_follows_the_flip(tmp_path):
    arguments = petsc.configure_arguments(
        source_dir=tmp_path,
        prefix=tmp_path,
        mpi_prefix=tmp_path,
        jobs=1,
        scalar_type=FLIPPED,
    )

    assert f"--with-scalar-type={FLIPPED}" in arguments
    assert f"PETSC_ARCH={petsc.build_arch(FLIPPED)}" in arguments


def test_the_two_variants_do_not_share_a_petsc_build_tree():
    """`PETSC_ARCH` is what keeps the object trees apart in one source dir."""
    assert petsc.build_arch("complex") != petsc.build_arch("real")


def test_the_built_petsc_is_judged_against_the_variant_asked_for():
    """The same install passes for one variant and fails for the other."""
    complex_build = COMPLEX_BUILD

    assert petsc.build_problem(complex_build, scalar_type="complex") is None
    problem = petsc.build_problem(complex_build, scalar_type="real")
    assert problem is not None
    assert "real-scalar one" in problem


def test_the_in_process_check_proves_the_variant_rather_than_complex():
    """`import_check` runs from three stages; it was the first to fail."""
    assert import_check.scalar_problem(float, variant="real") is None
    assert import_check.scalar_problem(complex, variant="complex") is None
    assert import_check.scalar_problem(complex, variant="real") is not None
    assert import_check.scalar_problem(float, variant="complex") is not None


@pytest.mark.parametrize("with_dolfinx", [False, True])
def test_the_summary_line_is_not_the_place_the_variant_is_decided(with_dolfinx):
    """Both stages print it, and the DOLFINx one also lists the feature names
    — `has_complex_ufcx_kernels` among them, which is the compiler's and not
    the variant's (see EXPECTED_FEATURES)."""
    line = import_check.summary(
        site=Path("/site"),
        mapped=["/venv/lib/libmpi.so.12"],
        variant="real",
        with_dolfinx=with_dolfinx,
    )

    assert "real scalars" in line
    assert "complex scalars" not in line


@pytest.mark.parametrize("variant", ["complex", "real"])
def test_a_prefix_is_claimed_by_one_variant_and_refuses_the_other(tmp_path, variant):
    prefix_module.claim_variant(tmp_path, variant)

    with pytest.raises(ValueError, match="BUILD_ROOT"):
        prefix_module.claim_variant(tmp_path, OTHER[variant])


def test_the_refusal_is_what_the_build_script_sees(tmp_path):
    """The shell runs the module, so the refusal has to be a non-zero exit."""
    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "wheelbuild.prefix",
            "--prefix",
            str(tmp_path),
            "--scalar-type",
            "complex",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        [
            sys.executable,
            "-m",
            "wheelbuild.prefix",
            "--prefix",
            str(tmp_path),
            "--scalar-type",
            "real",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert first.returncode == 0
    assert second.returncode == 1
    assert "complex" in second.stderr
