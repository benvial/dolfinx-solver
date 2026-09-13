"""What the build tooling lock has to be, and what it may not hold.

Every source archive the build compiles is pinned by content (ticket 10, spec
§10); the other thing a run downloads is the build venv, and until ticket 25 it
was a line of floors resolved fresh on the day. These are the rules that make
the two variants of one release provably the work of the same tooling: the lock
is complete and hashed, it is installed as such, and it does not hold a second
copy of a pin that lives in a driver.
"""

import re
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "wheelbuild" / "requirements.txt"
REQUIREMENTS_IN = REPO_ROOT / "wheelbuild" / "requirements.in"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build-wheel.sh"

#: What a requirement line of the lock looks like once its trailing
#: continuation is off: exactly a name and one version, which is all
#: `--require-hashes` accepts.
PINNED_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)$"
)

#: How many lines of the lock are the hand-written header. Everything the
#: header has to say — what the file is for, and the command that regenerates
#: it — is above the first pin, so this is only a bound on where to look.
HEADER_LINES = 40

#: The four the ticket singled out, because they are not build-only in the way
#: the rest are: two decide what the published wheel claims, and two are
#: compiled and imported against.
DECIDES_THE_ARTEFACT = ("auditwheel", "abi3audit", "numpy", "mpi4py")

#: The derived pins (CONTEXT.md), which must not appear in the lock: each is
#: computed at run time from the one driver that owns it, and a copy here
#: would be a second one to keep in step, away from its reasoning. The
#: upstream trio is three of the five.
DERIVED_PINS = (
    "nanobind",
    "mpich",
    "fenics-basix",
    "fenics-ffcx",
    "fenics-ufl",
)


class Pin(NamedTuple):
    """One requirement of the lock, as the compiled file spells it.

    Attributes:
        name: The distribution, lowercased.
        version: The single version it is pinned to.
        hashes: The `--hash=` lines under it, which is what
            `--require-hashes` checks a download against.
    """

    name: str
    version: str
    hashes: tuple[str, ...]


def read_lock(requirements: str) -> tuple[dict[str, Pin], list[str]]:
    """Read the lock the way pip does, and report what is not a pin.

    A compiled file is requirement lines at the left margin, each followed by
    its indented `--hash=` continuations and a `# via` provenance comment.

    Args:
        requirements: The file's text.

    Returns:
        The pins by name, and every left-margin line that is not one — which
        is what `--require-hashes` would refuse.
    """
    pins: dict[str, Pin] = {}
    hashes: dict[str, list[str]] = {}
    loose: list[str] = []
    current = ""
    for raw in requirements.splitlines():
        line = raw.removesuffix("\\").strip()
        if not line or line.startswith("#"):
            continue
        if raw.startswith((" ", "\t")):
            if line.startswith("--hash=") and current:
                hashes[current].append(line.removeprefix("--hash="))
            continue
        pinned = PINNED_LINE.match(line)
        if pinned is None:
            loose.append(line)
            continue
        current = pinned["name"].lower()
        hashes[current] = []
        pins[current] = Pin(current, pinned["version"], ())
    return {
        name: pin._replace(hashes=tuple(hashes[name])) for name, pin in pins.items()
    }, loose


@pytest.fixture(scope="module")
def requirements() -> str:
    return REQUIREMENTS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def pins(requirements) -> dict[str, Pin]:
    return read_lock(requirements)[0]


@pytest.fixture(scope="module")
def script() -> str:
    return BUILD_SCRIPT.read_text(encoding="utf-8")


def direct_requirements() -> list[str]:
    """The names this repository chose, read from the compiler's input.

    `requirements.in` is the hand-written half and carries floors, so a name
    is what stands before the first comparison operator.
    """
    text = REQUIREMENTS_IN.read_text(encoding="utf-8")
    return [
        re.split(r"[<>=!~]", line)[0].strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]


def test_every_requirement_is_pinned_to_one_version(requirements):
    """A floor in here is a build pip refuses, not a build that drifts."""
    loose = read_lock(requirements)[1]

    assert not loose, f"--require-hashes accepts no floors: {loose}"


def test_every_pin_carries_at_least_one_hash(pins):
    """The version says which release; the hash says which bytes."""
    unhashed = [name for name, pin in pins.items() if not pin.hashes]

    assert not unhashed


def test_every_hash_is_a_sha256(pins):
    """The one algorithm the sources are pinned with too (wheelbuild.sources)."""
    other = [
        digest
        for pin in pins.values()
        for digest in pin.hashes
        if not digest.startswith("sha256:")
    ]

    assert not other


def test_the_transitive_closure_is_in_the_file_not_only_the_names_we_chose(pins):
    """`--require-hashes` needs the whole tree, which is what compiling buys."""
    assert len(pins) > len(direct_requirements())


@pytest.mark.parametrize("name", direct_requirements())
def test_every_name_the_input_asks_for_is_in_the_compiled_file(name, pins):
    """An edit to requirements.in that was never compiled fails here."""
    assert name in pins, f"{name} is in requirements.in and not in the lock"


@pytest.mark.parametrize("name", DECIDES_THE_ARTEFACT)
def test_what_decides_the_artefact_is_pinned_rather_than_floored(name, pins):
    """The recorded decision of ticket 25, asserted rather than only written."""
    assert name in pins


@pytest.mark.parametrize("name", DERIVED_PINS)
def test_a_pin_that_lives_in_a_driver_is_not_copied_here(name, pins):
    assert name not in pins


def test_the_lock_says_how_it_is_regenerated(requirements):
    """It is generated, so the command that generates it travels with it."""
    header = "\n".join(requirements.splitlines()[:HEADER_LINES])

    assert "uv pip compile" in header
    assert "--generate-hashes" in header
    assert "requirements.in" in header


def test_the_resolution_is_made_for_the_interpreter_the_build_uses(requirements):
    """A lock resolved for anything else installs other wheels in the image."""
    assert "--python-version 3.12" in requirements
    assert "manylinux_2_34" in requirements


def test_the_build_installs_the_lock_and_requires_its_hashes(script):
    assert 'tooling_lock="$repo_root/wheelbuild/requirements.txt"' in script
    assert '--require-hashes -r "$tooling_lock"' in script


def test_the_build_installs_no_tooling_the_lock_does_not_hold(script, pins):
    """The floors the shell used to spell are gone, not merely supplemented."""
    environment = script[script.index('echo "==> build environment"') :]
    environment = environment[: environment.index('echo "==> pin check"')]
    hashed_install = environment[: environment.index("--require-hashes")]

    for floor in ('"cython>=3"', '"numpy>=2"', '"scikit-build-core>=0.11"'):
        assert floor not in environment, f"{floor} is still installed by hand"
    assert "--upgrade pip" not in hashed_install, "pip is pinned in the lock"
    assert "pip" in pins


def test_the_build_log_says_which_tooling_built_the_wheel(script):
    """Two variants of one release, and two logs that can be compared."""
    assert '"$venv/bin/pip" list --format=freeze' in script


def test_the_tooling_is_reported_after_the_derived_pins_are_installed(script):
    """Otherwise the report is missing nanobind, the trio and the mpich bound."""
    report = script.index('pip" list --format=freeze')

    assert script.index("NANOBIND_REQUIREMENT") < report
    assert script.index("upstream_trio_requirements") < report
    assert script.index("$mpich_requirement") < report
