"""What the build script's cache markers have to carry, read from the script.

`scripts/build-wheel.sh` decides "this stage is already built" against a warm
install prefix, and a marker that cannot notice an input has changed keeps a
stale build silently — which is a wrong wheel, not a slow one. The rules are
asserted against the script text because the stages themselves take hours and
the shell is where the markers are named.
"""

import re
from pathlib import Path

import pytest

from wheelbuild.mpich import MPICH_VERSION
from wheelbuild.petsc import SCALAR_TYPE

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build-wheel.sh"

#: Stamp assignments, as `<name>_stamp="$install_prefix/.<...>.installed"`.
STAMP = re.compile(r'^(?P<stage>\w+)_stamp="(?P<path>[^"]+)"$', re.MULTILINE)


@pytest.fixture(scope="module")
def script() -> str:
    return BUILD_SCRIPT.read_text()


@pytest.fixture(scope="module")
def stamps(script) -> dict[str, str]:
    return {match["stage"]: match["path"] for match in STAMP.finditer(script)}


def stage(script: str, heading: str) -> str:
    """The lines of one `==>` stage, up to the next one."""
    blocks = script.split('echo "==> ')
    matching = [block for block in blocks if block.startswith(heading)]
    assert matching, f"no stage starting {heading!r}"
    return matching[0]


def test_every_stage_that_can_be_skipped_is_skipped_on_a_stamp(stamps):
    """The MPICH stage joined the other six; a library file is not proof."""
    assert set(stamps) == {
        "mpich",
        "petsc",
        "slepc",
        "bindings",
        "adios2",
        "kahip",
        "dolfinx",
    }


def test_no_stage_is_cached_on_an_installed_library(script):
    """A library appears partway through `make install`; a stamp does not."""
    assert "libmpifort.so.12" not in script


def test_the_mpich_stamp_carries_the_release_it_was_built_from(stamps):
    assert stamps["mpich"] == "$install_prefix/.mpich-$mpich_version.installed"


def test_the_mpich_stamp_is_written_only_after_the_driver_returns(script):
    """`set -e` is what makes this an assertion: a failed build never stamps."""
    mpich = stage(script, "MPICH")
    build = mpich.index("--jobs")
    assert mpich.index('touch "$mpich_stamp"') > build


def test_the_scalar_type_is_read_from_the_petsc_driver(script):
    """One place to move when the real variant is built (spec §6)."""
    assert 'scalar_type="$(driver ' in script
    assert "from wheelbuild.petsc import SCALAR_TYPE" in script


def test_no_command_in_the_script_spells_the_scalar_type_itself(script):
    """A literal is what the driver's own flip could not have changed."""
    commands = [
        line
        for line in script.splitlines()
        if not line.lstrip().startswith("#")
        if SCALAR_TYPE in line.replace("--scalar-type", "")
    ]
    assert commands == []


def test_the_prefix_is_claimed_by_the_variant_before_anything_builds(script):
    """One prefix per variant; stamps are never pruned (ticket 17)."""
    layout = stage(script, "install prefix layout")

    assert '--scalar-type "$scalar_type"' in layout
    assert script.index('scalar_type="$(driver ') < script.index(
        'echo "==> install prefix layout"'
    )


def test_every_stamp_bound_to_petsc_names_the_release(stamps):
    """Anything built against something PETSc's configure produced.

    ADIOS2 is in the list for its HDF5: `-DHDF5_ROOT=$prefix` with
    `-DHDF5_PREFER_PARALLEL=ON` resolves the parallel HDF5 PETSc built, so a
    PETSc bump moves a library ADIOS2 is linked against.
    """
    bound = {stage for stage, path in stamps.items() if "$petsc_version" in path}
    assert bound == {"petsc", "slepc", "bindings", "adios2", "dolfinx"}


def test_every_stamp_carrying_petsc_scalars_names_the_scalar_type(stamps):
    """PetscScalar is baked into everything that links libpetsc, sonames and all.

    ADIOS2 is the one PETSc-bound stage that does not: it links the prefix's
    HDF5 and never libpetsc, and HDF5 is built the same way for either scalar
    type, so a flip would rebuild it for nothing.
    """
    for stage_name in ("petsc", "slepc", "bindings", "dolfinx"):
        assert "$scalar_type" in stamps[stage_name], f"{stage_name} is scalar-blind"
    assert "$scalar_type" not in stamps["adios2"]


def test_every_stamp_bound_to_mpich_names_the_release(stamps):
    """Only the stages a bump makes produce something different name MPICH.

    ADIOS2 and KaHIP are compiled with the prefix's `mpicc`/`mpicxx`, and
    ticket 18 keyed them on the release for the exercise of it: their rebuilds
    are minutes, so naming the release costs nothing against the chance that a
    bump moved something they were linked against.

    The four stages that do not name it are compiled the same way, and ticket
    19 decided deliberately to leave them: the wheel's whole MPI story is that
    every vendored library asks the loader for `libmpi.so.12` and gets the
    PyPI mpich wheel's copy — a different build from the one in the build
    prefix — so being linked against one 5.x libmpi rather than another is not
    a property the wheel preserves or needs (spec §5, ADR-0001). PETSc's
    rebuild is hours, which is why the insurance ADIOS2 and KaHIP take is not
    worth taking here. The stage comments in the script carry the same reason.
    """
    bound = {stage for stage, path in stamps.items() if "$mpich_version" in path}
    assert bound == {"mpich", "adios2", "kahip"}


@pytest.mark.parametrize(
    "heading",
    ["PETSc (", "SLEPc (", "petsc4py + slepc4py", "DOLFINx ("],
)
def test_a_stage_that_does_not_name_mpich_says_why_not(script, heading):
    """The asymmetry with ADIOS2 and KaHIP is a decision, not an oversight."""
    block = stage(script, heading)
    assert "ADR-0001" in block, f"{heading} does not say why it may skip a relink"


def test_the_adios2_stamp_names_the_petsc_and_the_mpich_it_was_built_against(stamps):
    assert stamps["adios2"] == (
        "$install_prefix/.adios2-$adios2_version"
        "-petsc-$petsc_version-mpich-$mpich_version.installed"
    )


def test_the_kahip_stamp_names_the_mpich_it_was_built_against(stamps):
    assert stamps["kahip"] == (
        "$install_prefix/.kahip-$kahip_version-mpich-$mpich_version.installed"
    )


def test_the_petsc_driver_is_passed_the_scalar_type_on_both_paths(script):
    """Including `--validate-only`, which otherwise checks against its default."""
    petsc = stage(script, "PETSc")
    assert petsc.count('--scalar-type "$scalar_type"') == 2


def test_the_mpich_version_the_stamp_names_is_the_driver_s(script):
    """The shell reads it; nothing in the script holds a second copy."""
    assert MPICH_VERSION not in script
    assert "from wheelbuild.mpich import MPICH_VERSION" in script
