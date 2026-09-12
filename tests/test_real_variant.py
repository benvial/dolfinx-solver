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

import importlib
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from wheelbuild import assemble, import_check, notices, petsc
from wheelbuild import prefix as prefix_module
from wheeltest import demos, smoke, suite

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


#: The modules that read the variant at import time, so a flip only reaches
#: them through a re-import. `wheeltest.suite` reads `wheeltest.smoke`, so it
#: is reloaded after it.
VARIANT_MODULES = (smoke, demos, suite, notices)


@contextmanager
def built_for(variant: str):
    """Re-import the variant-reading modules as if the driver declared `variant`.

    The flip is a one-line edit to `wheelbuild.petsc.SCALAR_TYPE` followed by
    a build, which no test can do. What can be done is what the modules do
    with the value, which is read once at import; so the value is replaced and
    the modules are re-imported, then both are put back.
    """
    original = petsc.SCALAR_TYPE
    petsc.SCALAR_TYPE = variant
    try:
        for module in VARIANT_MODULES:
            importlib.reload(module)
        yield
    finally:
        petsc.SCALAR_TYPE = original
        for module in VARIANT_MODULES:
            importlib.reload(module)


def notice_text() -> str:
    """Render the notice file's variant-carrying parts, as a wheel ships them."""
    component = next(item for item in notices.COMPONENTS if item.name == "PETSc")
    return notices.render(
        [(component, ((Path(f"petsc-{petsc.PETSC_VERSION}/LICENSE"), "licence"),))],
        distribution_version="0.0.0",
    )


def test_the_smoke_stage_solves_the_variants_own_problem():
    """A real stack cannot represent the complex problem's boundary data, so
    the flip has to move the problem and not only the words about it."""
    assert smoke.problem_for("complex").solve is smoke.helmholtz
    assert smoke.problem_for("real").solve is smoke.poisson


def test_the_smoke_stage_judges_the_solution_against_the_variants_dtype():
    """The answer's own dtype, which is `PetscScalar` read off a result."""
    assert smoke.dtype_problem("complex128", variant="complex") is None
    assert smoke.dtype_problem("float64", variant="real") is None
    assert smoke.dtype_problem("float64", variant="complex") is not None
    assert smoke.dtype_problem("complex128", variant="real") is not None


def test_the_demo_subset_follows_the_flip():
    """Upstream's complex demos interpolate `exp(1j...)` without asking what
    `PetscScalar` is: under a real build numpy drops the imaginary part and
    the demo exits zero, so leaving them in would be a stage that passes for
    the wrong reason."""
    with built_for(FLIPPED):
        chosen = demos.DEMOS

    assert chosen == demos.DEMO_SUBSETS[FLIPPED]
    assert chosen != demos.DEMO_SUBSETS[OTHER[FLIPPED]]


def test_the_suites_report_follows_the_flip():
    """The `proves` lines are what a user reads when the suite passes."""
    with built_for(FLIPPED):
        report = " ".join(stage.proves for stage in suite.STAGES)

    assert FLIPPED in report
    assert OTHER[FLIPPED] not in report


def test_the_notice_text_follows_the_flip():
    """The notice file ships inside the wheel, so a wrong word is published."""
    with built_for(FLIPPED):
        text = notice_text()

    assert f"Built for {FLIPPED} scalars" in text
    assert OTHER[FLIPPED] not in text


def test_the_notice_names_the_distribution_the_assembler_builds():
    """The two spellings of one name (spec §6). This fails the moment the
    driver flips, because the distribution name is still a literal in
    `wheelbuild.assemble` and in the packaging metadata — which is ticket 21's
    half of the flip, and this is the reminder."""
    assert notices.DISTRIBUTION.replace("-", "_") == assemble.DISTRIBUTION
