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

from wheelbuild import dolfinx as dolfinx_module
from wheelbuild import mpich as mpich_module
from wheelbuild import petsc as petsc_module
from wheelbuild.mpich import MPICH_VERSION
from wheelbuild.petsc import SCALAR_TYPE

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build-wheel.sh"

#: Stamp assignments, as `<name>_stamp="$install_prefix/.<...>.installed"`.
STAMP = re.compile(r'^(?P<stage>\w+)_stamp="(?P<path>[^"]+)"$', re.MULTILINE)

#: The one stamp that is not a build stage's, and so is not swept by the rules
#: below. The build venv compiles nothing and installs into no prefix: its
#: stamp lives in the venv and carries the lock's digest rather than a release
#: (ticket 25). The test at the bottom is what keeps this list at one.
TOOLING_STAMP = "tooling"


@pytest.fixture(scope="module")
def script() -> str:
    return BUILD_SCRIPT.read_text()


@pytest.fixture(scope="module")
def all_stamps(script) -> dict[str, str]:
    return {match["stage"]: match["path"] for match in STAMP.finditer(script)}


@pytest.fixture(scope="module")
def stamps(all_stamps) -> dict[str, str]:
    return {stage: path for stage, path in all_stamps.items() if stage != TOOLING_STAMP}


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


#: `fetch_source <url> <dir> <digest>` calls, as they appear in the script.
FETCH = re.compile(
    r'^fetch_source "\$(?P<url>\w+)" "\$\w+" "\$(?P<digest>\w+)"$', re.MULTILINE
)


def test_every_source_archive_is_fetched_with_a_digest_to_check_it_against(script):
    """Six archives become vendored binaries; HTTPS pins none of them.

    A tarball re-rolled upstream under the same name satisfies the version
    pin and changes what is inside the published wheel, so every fetch names
    a SHA-256 as well as a URL (ticket 10, spec §10).
    """
    fetched = {match["url"].removesuffix("_url") for match in FETCH.finditer(script)}

    assert fetched == {
        "mpich",
        "petsc",
        "slepc",
        "adios2",
        "kahip",
        "dolfinx",
    }
    # Every call matched: one that passed two arguments would be a stage
    # extracting an archive nothing had checked.
    assert script.count('fetch_source "') == len(fetched)


def test_every_digest_is_read_from_the_driver_that_names_the_url(script):
    """One place to move when a version moves, and it is not the shell."""
    for match in FETCH.finditer(script):
        stage_name = match["url"].removesuffix("_url")
        constant = f"{stage_name.upper()}_SHA256"
        assert match["digest"] == f"{stage_name}_sha256"
        assert f"import {constant}; print({constant})" in script


def test_nothing_is_unpacked_before_its_digest_has_been_checked(script):
    """`tar` runs after the check, on the run that extracts, every time."""
    definition = script[script.index("fetch_source() {") :]
    definition = definition[: definition.index("\n}\n")]

    assert definition.index("wheelbuild.sources") < definition.index("tar -xzf")


def test_the_shell_fetches_through_the_module_that_holds_the_refusal_rule(script):
    """One implementation of the rule, not one per language (ticket 28).

    The `curl` this replaced was unconditional, so a refused archive was
    downloaded again on every run and a mismatch that cleared upstream between
    two runs passed on the second. `wheelbuild.sources.fetch` is where the
    rejection marker that refuses to do that lives.
    """
    definition = script[script.index("fetch_source() {") :]
    definition = definition[: definition.index("\n}\n")]

    assert "curl" not in definition
    assert "curl -" not in script, "a download the refusal rule does not cover"
    assert '--url "$url"' in definition


def test_the_archive_is_fetched_before_the_tree_it_replaces_is_deleted(script):
    """A refusal is never retried now, so a deleted tree would not come back.

    The same order `wheeltest.demos.fetch` took for the demo cache (ticket
    26): obtain the replacement first, because a fetch that cannot produce the
    pinned bytes leaves a run that still has a source tree to say it about.
    """
    definition = script[script.index("fetch_source() {") :]
    definition = definition[: definition.index("\n}\n")]

    assert definition.index("wheelbuild.sources") < definition.index("rm -rf")
    assert definition.index("rm -rf") < definition.index("tar -xzf")


def test_a_refused_archive_stops_the_build_without_relying_on_set_e(script):
    """`set -e` is switched off for a function body called in a condition.

    Nothing calls `fetch_source` that way today, and a check that unpacks the
    archive anyway if someone ever does is not a check.
    """
    definition = script[script.index("fetch_source() {") :]
    definition = definition[: definition.index("\n}\n")]

    assert "exit 1" in definition
    assert definition.index("exit 1") < definition.index("tar -xzf")


def test_the_extraction_marker_records_the_digest_it_was_extracted_from(script):
    """A digest that moves without the version has to re-extract, not reuse."""
    definition = script[script.index("fetch_source() {") :]
    definition = definition[: definition.index("\n}\n")]

    assert '"$(cat "$marker")" != "$expected"' in definition
    assert 'printf \'%s\\n\' "$expected" >"$marker"' in definition


def test_what_an_empty_marker_costs_is_written_where_the_marker_is(script):
    """Accepted rather than mitigated (ticket 27), so it is said out loud.

    Markers written before the digests existed are bare `touch` files, and the
    run that meets one re-extracts every source tree — including PETSc's,
    which is also its build tree. The decision not to hash around that only
    holds if the next reader finds the reasoning instead of the symptom.
    """
    preamble = script[: script.index("fetch_source() {")]
    reason = preamble[preamble.rindex("# Download, verify") :]

    assert "ticket 27" in reason
    assert "externalpackages" in reason, "the cost is PETSc's build tree, unnamed"


def test_the_trio_pins_are_checked_against_the_verified_source_tree(script):
    """Upstream's own file, out of bytes this build pinned (ticket 10)."""
    dolfinx = stage(script, "DOLFINx (")
    check = dolfinx.index("wheelbuild.pin_check")

    assert dolfinx.index("fetch_source") < check
    assert '--upstream-pyproject "$dolfinx_source/python/pyproject.toml"' in dolfinx


def test_no_digest_is_spelled_in_the_shell(script):
    """The same rule the versions follow: the drivers hold the constants."""
    for constant in (
        mpich_module.MPICH_SHA256,
        petsc_module.PETSC_SHA256,
        dolfinx_module.DOLFINX_SHA256,
    ):
        assert constant not in script


CONTAINER_SCRIPT = REPO_ROOT / "scripts" / "build-in-container.sh"
VENV_SCRIPT = REPO_ROOT / "scripts" / "_wheel_venv.sh"


@pytest.fixture(scope="module")
def container_script() -> str:
    return CONTAINER_SCRIPT.read_text()


@pytest.fixture(scope="module")
def venv_script() -> str:
    return VENV_SCRIPT.read_text()


def test_a_local_build_caches_each_variant_in_its_own_directory(container_script):
    """One prefix holds one scalar type and refuses the other (ticket 17), so
    a flip on a developer's machine has to land in a second directory rather
    than in a refusal they have to read a ticket to understand."""
    assert 'cache_dir="${BUILD_CACHE:-$repo_root/.build-cache-$scalar_type}"' in (
        container_script
    )


def test_the_local_entry_points_read_the_variant_from_the_driver(
    container_script, venv_script
):
    """The same single place the build script reads it from: a literal here
    could not follow the environment the matrix sets."""
    for text in (container_script, venv_script):
        assert "from wheelbuild.petsc import SCALAR_TYPE" in text
        assert SCALAR_TYPE not in re.sub(
            r"^#.*$", "", text, flags=re.MULTILINE
        ).replace("SCALAR_TYPE", "")


def test_the_container_is_told_which_variant_it_is_building(container_script):
    """The host resolves it; the container's drivers have to see the same
    value, or the build in it would produce the default variant."""
    assert f'-e {petsc_module.SCALAR_TYPE_VARIABLE}="$scalar_type"' in container_script


def test_the_suite_defaults_to_the_wheelhouse_its_variants_build_left(venv_script):
    """Two build roots, two wheelhouses: a default naming neither would test
    whichever wheel happened to be there."""
    assert ".build-cache-$scalar_type/wheelhouse" in venv_script
    assert ".build-cache-$scalar_type/wheeltest" in venv_script


def test_the_only_stamp_outside_the_install_prefix_is_the_build_venvs(all_stamps):
    """A build stage's stamp lives with what it installed, and is swept above.

    The build venv's does not, because the venv is tooling rather than a stage
    (CONTEXT.md, *Stage*). Anything else appearing out here is a stage whose
    stamp the rules above stopped seeing.
    """
    outside = {
        stage: path
        for stage, path in all_stamps.items()
        if not path.startswith("$install_prefix/")
    }

    assert outside == {TOOLING_STAMP: "$venv/.tooling-lock"}


def test_a_restored_build_venv_is_rebuilt_when_the_lock_moves(script):
    """`pip install` never prunes, so layering leaves what the lock dropped."""
    environment = stage(script, "build environment")

    assert '"$(cat "$tooling_stamp" 2>/dev/null)" != "$tooling_digest"' in environment
    assert 'rm -rf "${venv:?}"' in environment


def test_the_build_venv_stamp_is_written_only_after_every_install(script):
    """The same rule the stage stamps follow: installed *and* proved first.

    The derived pins are installed after the lock, so a stamp written between
    them would adopt a venv whose last run died halfway through the three.
    """
    written = script.index('>"$tooling_stamp"')

    assert script.index("--require-hashes") < written
    assert script.index("$mpich_requirement") < written
    assert script.index('pip" list --format=freeze') < written
    assert written < script.index('echo "==> pin check"')


#: The stages whose output is compiled against something the build venv holds:
#: petsc4py's and slepc4py's extensions take their headers from the lock's
#: numpy and are generated by its Cython and setuptools, and the DOLFINx
#: bindings are configured by its scikit-build-core and compiled against
#: mpi4py's `include`. The other five compile against the prefix and the
#: container's compilers only (ticket 29).
COMPILED_AGAINST_THE_LOCK = {"bindings", "dolfinx"}


def test_every_stage_compiled_against_the_lock_names_it(stamps):
    """A lock bump reinstalls the venv; without this it re-checks these two.

    The wheel would then hold extensions compiled against the previous numpy
    while `pip list --format=freeze` in the same log reports the new one —
    the reproducibility claim ticket 25 exists to make, failing quietly on
    exactly the path CI takes, since the superbuild cache's `restore-keys`
    deliberately accepts the entry the exact key just missed.
    """
    bound = {stage for stage, path in stamps.items() if "$tooling_id" in path}

    assert bound == COMPILED_AGAINST_THE_LOCK


def test_the_tooling_the_stamps_name_is_the_lock_the_venv_was_built_from(script):
    """One digest, read once: a second `sha256sum` could name another file."""
    environment = stage(script, "build environment")

    assert 'tooling_digest="$(sha256sum "$tooling_lock" | cut -d\' \' -f1)"' in (
        environment
    )
    assert 'tooling_id="${tooling_digest:0:12}"' in environment
    assert script.count("sha256sum") == 1


def test_the_tooling_id_is_read_before_the_first_stamp_names_it(script):
    """It is assigned in the build environment stage, which runs before all."""
    assigned = script.index('tooling_id="${tooling_digest:0:12}"')

    for stage_name in COMPILED_AGAINST_THE_LOCK:
        assert assigned < script.index(f"{stage_name}_stamp=")


@pytest.mark.parametrize("heading", ["MPICH", "PETSc (", "SLEPc (", "ADIOS2", "KaHIP"])
def test_a_stage_that_does_not_name_the_tooling_says_why_not(script, heading):
    """The asymmetry with the two bindings stages is a decision, not an
    oversight: these five are compiled by the container's own toolchain
    against the install prefix, and nothing the lock holds is an input to
    what they produce."""
    block = stage(script, heading)

    assert "ticket 29" in block, f"{heading} does not say why a lock bump is not its"
