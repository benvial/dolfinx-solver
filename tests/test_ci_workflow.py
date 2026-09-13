"""What the wheels workflow has to say, read from the workflow itself.

The workflow is the only place some of this project's contracts are enforced:
the pin checks run there, the superbuild cache key is what decides whether a
push recompiles for hours or re-validates in a minute, and the platform tag on
the wheel is a claim about machines other than the one that built it. None of
that is reachable from a unit test the way code is, so it is asserted against
the file — the same approach `test_build_script.py` takes to the cache stamps.
"""

from pathlib import Path

import pytest
import yaml

from wheelbuild.petsc import SCALAR_TYPE_VARIABLE, SCALAR_TYPES
from wheelbuild.version import DOLFINX_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "wheels.yml"

#: The image the wheel's platform tag is a promise about (spec §2, §8).
MANYLINUX_IMAGE = "quay.io/pypa/manylinux_2_34_x86_64"

#: Every input that can change what a build stage should produce. A cache key
#: missing one of these restores a tree whose stamps say "already built" for
#: something that is no longer what the drivers would build.
CACHE_KEY_INPUTS = (
    "wheelbuild/**/*.py",
    # The build venv's hashed tooling (ticket 25). Not covered by the glob
    # above, and a bumped auditwheel or numpy is a different wheel out of the
    # same sources.
    "wheelbuild/requirements.txt",
    "scripts/build-wheel.sh",
    "dolfinx_solver/_version.py",
    "pyproject.toml",
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def triggers(workflow) -> dict:
    # PyYAML resolves a bare `on:` key to the boolean True, which is the one
    # place YAML 1.1 and GitHub's schema disagree.
    return workflow[True]


def _steps(workflow: dict, job: str) -> list[dict]:
    return workflow["jobs"][job]["steps"]


def _step(workflow: dict, job: str, name: str) -> dict:
    for step in _steps(workflow, job):
        if step.get("name") == name:
            return step
    raise AssertionError(f"the {job} job has no step named {name!r}")


def _run_text(workflow: dict, job: str) -> str:
    return "\n".join(step.get("run", "") for step in _steps(workflow, job))


def test_every_trigger_the_spec_names_is_wired(triggers):
    """Spec §9: push to main, `v*` tags, PRs, manual dispatch."""
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["push"]["tags"] == ["v*"]
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers


def test_runs_are_grouped_per_ref(workflow):
    assert "github.ref" in workflow["concurrency"]["group"]


def test_a_build_is_cancelled_only_where_losing_its_cache_costs_nothing(workflow):
    """The cache is written by the job's post-step, so cancelling a cold
    superbuild throws away the hours it spent. Two pushes to main would then
    leave main permanently cold; pull requests are cheap to redo."""
    cancel = workflow["concurrency"]["cancel-in-progress"]

    assert "pull_request" in str(cancel)
    assert cancel is not True


def test_the_cache_does_not_carry_what_is_rebuilt_every_run(workflow):
    """`!` lines under a cached directory exclude nothing: actions/cache
    resolves `path` with implicit descendants off. Deleting is what works."""
    cache = _step(workflow, "wheel", "Restore the superbuild cache")["with"]

    assert "!" not in cache["path"]

    dropped = _step(workflow, "wheel", "Drop what the cache must not carry")

    assert "$BUILD_ROOT/assemble" in dropped["run"]
    assert "$BUILD_ROOT/wheelhouse" in dropped["run"]
    assert dropped["if"] == "always()"


def test_the_wheel_is_collected_before_the_cached_wheelhouse_is_dropped(workflow):
    names = [step.get("name") for step in _steps(workflow, "wheel")]

    assert names.index("Collect the wheel") < names.index(
        "Drop what the cache must not carry"
    )


def test_the_cheap_checks_gate_the_expensive_build(workflow):
    assert workflow["jobs"]["wheel"]["needs"] == "checks"
    assert workflow["jobs"]["tests"]["needs"] == "wheel"


def test_the_checks_job_lints_and_type_checks_every_package(workflow):
    commands = _run_text(workflow, "checks")

    assert "ruff check ." in commands
    assert "ruff format --check ." in commands
    assert "mypy dolfinx_solver wheelbuild wheeltest" in commands
    assert "pytest" in commands


def test_the_pin_check_compares_against_upstreams_own_pins(workflow):
    """Without `--fetch-upstream` the trio half of the check does not run."""
    step = _step(workflow, "checks", "Pin check")

    assert "wheelbuild.pin_check" in step["run"]
    assert "--fetch-upstream" in step["run"]


def test_the_tag_check_runs_on_tag_refs_and_only_there(workflow):
    step = _step(workflow, "checks", "Tag check")

    assert "wheelbuild.tag_check" in step["run"]
    assert step["if"] == "startsWith(github.ref, 'refs/tags/v')"


def test_the_wheel_is_compiled_in_the_image_its_tag_promises(workflow):
    assert workflow["jobs"]["wheel"]["container"]["image"] == MANYLINUX_IMAGE


@pytest.mark.parametrize("hashed", CACHE_KEY_INPUTS)
def test_the_cache_key_covers_every_input_to_a_build_stage(workflow, hashed):
    key = _step(workflow, "wheel", "Restore the superbuild cache")["with"]["key"]

    assert hashed in key


def test_the_cache_key_names_the_scalar_type_the_variant_is_built_for(workflow):
    """Ticket 12, then 21: two jobs build two variants, and neither may
    restore the other's prefix — the stamps in it say "already built" for a
    stack of the wrong scalar type (spec §6). The matrix value is in the key
    rather than a literal, so the job that builds a variant is the job that
    names it."""
    cache = _step(workflow, "wheel", "Restore the superbuild cache")["with"]

    assert "matrix.scalar-type" in cache["key"]
    assert "matrix.scalar-type" in cache["restore-keys"]
    for variant in SCALAR_TYPES:
        assert variant not in cache["key"]


def test_the_wheel_job_builds_every_variant_the_drivers_name(workflow):
    """Ticket 21: one workflow, both distributions (spec §6). A third variant
    added to the drivers is one the workflow has to grow a job for."""
    matrix = workflow["jobs"]["wheel"]["strategy"]["matrix"]["scalar-type"]

    assert matrix == list(SCALAR_TYPES)
    assert workflow["jobs"]["wheel"]["strategy"]["fail-fast"] is False


def test_the_variant_a_job_builds_is_declared_in_one_place(workflow):
    """`wheelbuild.petsc` resolves this variable once, and everything that
    reads the scalar type reads what it resolved — the configure line, the
    cache stamps, the prefix claim, the distribution name and the notices."""
    for job in ("wheel", "tests"):
        assert (
            workflow["jobs"][job]["env"][SCALAR_TYPE_VARIABLE]
            == "${{ matrix.scalar-type }}"
        )


def test_each_variant_builds_in_its_own_build_root(workflow):
    """`wheelbuild.prefix.claim_variant` refuses a prefix built for the other
    one (ticket 17), so sharing a directory is a failed job rather than a
    stale build — and the cached tree is the build root itself."""
    build_root = workflow["jobs"]["wheel"]["env"]["BUILD_ROOT"]
    cache = _step(workflow, "wheel", "Restore the superbuild cache")["with"]

    assert "matrix.scalar-type" in build_root
    assert cache["path"] == "${{ env.BUILD_ROOT }}"


def test_an_older_cache_entry_is_accepted_when_a_driver_changes(workflow):
    """The stamps inside decide what rebuilds; a warm ccache is worth keeping."""
    cache = _step(workflow, "wheel", "Restore the superbuild cache")["with"]
    prefix = cache["restore-keys"].strip()

    assert cache["key"].startswith(prefix)


def test_the_published_wheel_comes_from_the_build_roots_wheelhouse(workflow):
    """The install prefix holds another one, of artefacts that never ship."""
    collected = _step(workflow, "wheel", "Collect the wheel")["run"]

    assert '"$BUILD_ROOT"/wheelhouse/*.whl' in collected
    assert "install" not in collected


def test_the_wheel_is_uploaded_and_a_missing_one_fails_the_job(workflow):
    for step in _steps(workflow, "wheel"):
        if str(step.get("uses", "")).startswith("actions/upload-artifact"):
            assert step["with"]["if-no-files-found"] == "error"
            return
    raise AssertionError("the wheel job uploads nothing")


def test_each_variants_wheel_is_its_own_artefact(workflow):
    """Two jobs upload, and an artifact name is unique per workflow run: one
    name would be the second job failing on a name the first one took."""
    uploaded = next(
        step["with"]["name"]
        for step in _steps(workflow, "wheel")
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    )
    downloaded = next(
        step["with"]["name"]
        for step in _steps(workflow, "tests")
        if str(step.get("uses", "")).startswith("actions/download-artifact")
    )

    assert "matrix.scalar-type" in uploaded
    assert downloaded == uploaded


def test_every_variant_is_tested_on_every_interpreter(workflow):
    """The two axes are independent: one abi3 wheel per variant, three
    interpreters each."""
    matrix = workflow["jobs"]["tests"]["strategy"]["matrix"]

    assert matrix["scalar-type"] == list(SCALAR_TYPES)
    assert len(matrix["python-version"]) == 3


def test_the_wheel_is_tested_outside_the_image_that_built_it(workflow):
    """manylinux_2_34 is a claim about other machines (spec §9, ticket 07)."""
    assert "container" not in workflow["jobs"]["tests"]


def test_the_matrix_is_every_interpreter_the_abi3_wheel_claims(workflow):
    """One wheel, three interpreters — the classifiers say which."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    claimed = [
        line.split("::")[-1].strip().rstrip('",')
        for line in pyproject.splitlines()
        if "Programming Language :: Python :: 3." in line
    ]
    matrix = workflow["jobs"]["tests"]["strategy"]["matrix"]["python-version"]

    assert matrix == claimed


def test_one_interpreter_failing_does_not_hide_the_others(workflow):
    assert workflow["jobs"]["tests"]["strategy"]["fail-fast"] is False


def test_the_suite_runs_through_the_script_the_repository_ships(workflow):
    assert "scripts/test-wheel.sh" in _run_text(workflow, "tests")


def test_the_tests_job_sets_no_pythonpath():
    """Ticket 07: `scripts/_wheel_venv.sh` sets it, and a second value on top
    is how the repository's own dolfinx_solver comes to shadow the payload."""
    assert "PYTHONPATH" not in WORKFLOW.read_text(encoding="utf-8").replace(
        "# PYTHONPATH", ""
    )


def test_the_suite_keeps_its_scratch_out_of_the_checkout(workflow):
    """On the step rather than the job: `runner` is not a context a job-level
    `env:` may read, and GitHub rejects the workflow file outright for it."""
    step = _step(workflow, "tests", "Run the wheel test suite")
    work_dir = step["env"]["WORK_DIR"]

    assert "runner.temp" in work_dir
    assert "github.workspace" not in work_dir


def test_no_job_level_env_reads_a_context_github_refuses_there(workflow):
    """The failure this cost a run: a workflow that parses locally and is
    rejected before any job starts, with no job to read a log from."""
    for job in workflow["jobs"].values():
        for value in job.get("env", {}).values():
            assert "runner." not in str(value)


def test_the_demo_sources_are_cached_on_the_release_they_come_from(workflow):
    """One immutable upstream tag per release, downloaded once."""
    for step in _steps(workflow, "tests"):
        if str(step.get("uses", "")).startswith("actions/cache"):
            assert step["with"]["path"].endswith("demo-source")
            assert "dolfinx_solver/_version.py" in step["with"]["key"]
            return
    raise AssertionError("the tests job downloads the demo sources every run")


def test_the_workflow_does_not_publish_anything(workflow):
    """Spec §9 gives the upload its own workflow; a tag stops at a tested wheel.

    Trusted publishing turns on an OIDC token, so the absence of that
    permission is what makes a publishing step impossible here rather than
    merely absent.
    """
    assert "id-token" not in workflow["permissions"]
    assert workflow["permissions"]["contents"] == "read"

    for job in workflow["jobs"].values():
        for step in job["steps"]:
            assert "pypi" not in str(step.get("uses", "")).lower()


def test_the_mirrored_release_is_the_one_the_demo_cache_is_keyed_on():
    """A sanity tie: the version module the key hashes is where it comes from."""
    assert DOLFINX_VERSION in (REPO_ROOT / "dolfinx_solver" / "_version.py").read_text(
        encoding="utf-8"
    )


def test_one_variant_failing_to_build_still_tests_the_other(workflow):
    """`needs` waits for the whole matrix and skips dependents when any leg
    fails, so the wheel job's fail-fast: false would buy nothing: the variant
    that did build would go untested (ticket 21)."""
    tests = workflow["jobs"]["tests"]

    assert tests["needs"] == "wheel"
    assert "cancelled()" in tests["if"]
