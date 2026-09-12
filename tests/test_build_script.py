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


def test_every_stamp_bound_to_petsc_names_the_scalar_type(stamps):
    """PetscScalar is baked into everything below PETSc, sonames and all."""
    bound = {stage: path for stage, path in stamps.items() if "$petsc_version" in path}
    assert set(bound) == {"petsc", "slepc", "bindings", "dolfinx"}
    for stage_name, path in bound.items():
        assert "$scalar_type" in path, f"{stage_name} stamp is scalar-blind"


def test_the_petsc_driver_is_passed_the_scalar_type_on_both_paths(script):
    """Including `--validate-only`, which otherwise checks against its default."""
    petsc = stage(script, "PETSc")
    assert petsc.count('--scalar-type "$scalar_type"') == 2


def test_the_mpich_version_the_stamp_names_is_the_driver_s(script):
    """The shell reads it; nothing in the script holds a second copy."""
    assert MPICH_VERSION not in script
    assert "from wheelbuild.mpich import MPICH_VERSION" in script
