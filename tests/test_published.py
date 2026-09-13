"""The post-publish check's judgements, without an index to publish to.

What this stage does that no other run can — install the three distributions
by name and watch them resolve each other — needs a real upload. What it
decides once they are installed does not, and that is what is asserted here:
which variant a `pip install dolfinx-solver` has to have brought, where a dry
run's dependencies come from, and that an index which has not caught up yet
is waited for rather than reported as a broken release.
"""

import subprocess

import pytest

from wheelbuild import publish
from wheelbuild.petsc import DEFAULT_SCALAR_TYPE, SCALAR_TYPES
from wheelbuild.version import __version__
from wheeltest import published


def test_the_meta_package_resolving_to_the_variant_it_pins_is_the_pass():
    installed = [published.PINNED_DISTRIBUTION, "numpy", "mpich"]

    assert published.resolution_problem(installed) is None


def test_the_pinned_variant_is_the_one_the_meta_package_actually_pins():
    """Read out of `meta/pyproject.toml`, so flipping that pin moves what
    this check expects rather than leaving it asserting the old variant
    (spec §1, §6)."""
    assert f"dolfinx-solver-{DEFAULT_SCALAR_TYPE}" == published.PINNED_DISTRIBUTION
    assert published.PINNED_DISTRIBUTION in published.VARIANT_DISTRIBUTIONS
    assert published.PINNED_SCALAR_TYPE in SCALAR_TYPES
    assert len(published.VARIANT_DISTRIBUTIONS) == len(SCALAR_TYPES)


def test_a_meta_install_that_brought_no_binaries_is_reported():
    """The meta-package ships no files; an install of it that resolved
    nothing is an install with no DOLFINx in it."""
    problem = published.resolution_problem(["numpy", "mpi4py"])

    assert problem is not None
    assert "no variant wheel" in problem


def test_a_meta_install_that_brought_the_other_variant_is_reported():
    other = next(
        name
        for name in published.VARIANT_DISTRIBUTIONS
        if name != published.PINNED_DISTRIBUTION
    )
    problem = published.resolution_problem([other])

    assert problem is not None
    assert other in problem
    assert published.PINNED_DISTRIBUTION in problem


def test_an_environment_holding_both_variants_is_not_a_resolution():
    problem = published.resolution_problem(published.VARIANT_DISTRIBUTIONS)

    assert problem is not None


def test_the_release_is_asked_for_by_exact_version():
    """The index may already hold a later release; this check is about the
    one that was just uploaded."""
    assert published.requirements(["dolfinx-solver"]) == [
        f"dolfinx-solver=={__version__}"
    ]


def test_a_real_publish_installs_from_the_default_index():
    assert published.index_arguments("pypi") == []


def test_a_dry_run_falls_back_to_where_the_dependencies_actually_are():
    """TestPyPI holds this project's uploads and nothing else: mpich, mpi4py,
    numpy and the upstream trio are still on PyPI."""
    arguments = published.index_arguments("testpypi")

    assert "--index-url" in arguments
    assert "https://test.pypi.org/simple/" in arguments
    assert publish.FALLBACK_SIMPLE_URL in arguments


def test_the_release_itself_is_asked_of_one_index_with_no_fallback():
    """pip treats --index-url and --extra-index-url as one namespace, so a
    dry run allowed both could resolve our own distributions from PyPI and
    report success about a release TestPyPI never received."""
    arguments = published.strict_index_arguments("testpypi")

    assert arguments == ["--index-url", "https://test.pypi.org/simple/"]
    assert publish.FALLBACK_SIMPLE_URL not in arguments


@pytest.mark.parametrize(
    "arguments", [published.index_arguments, published.strict_index_arguments]
)
def test_an_index_this_project_does_not_publish_to_is_refused(arguments):
    with pytest.raises(ValueError, match="staging"):
        arguments("staging")


def test_the_stages_are_the_suites_own_rather_than_a_second_definition():
    """The import check is the one the build runs against its staging site,
    and the smoke test is told which variant to expect rather than reading
    the environment — what is under test is the one the pin resolved to."""
    modules = [module for module, _ in published.STAGES]

    assert modules == ["wheelbuild.import_check", "wheeltest.smoke"]
    assert published.PINNED_SCALAR_TYPE in dict(published.STAGES)["wheeltest.smoke"]


def test_every_distribution_is_asked_of_the_index(monkeypatch, tmp_path):
    """Not only the one the first venv installs: a release is all three, and
    a meta-package served ahead of the variant it pins would install from
    whatever the fallback index had."""
    commands = []

    monkeypatch.setattr(
        published, "check_call", lambda command, **_: commands.append(command)
    )
    published.wait_for_release(tmp_path / "served", version="1.0")

    assert "download" in commands[0]
    for name in publish.DISTRIBUTIONS:
        assert f"{name}==1.0" in commands[0]


def test_an_index_that_has_not_caught_up_is_waited_for(monkeypatch, tmp_path):
    """An upload is accepted before it is installable, and a check that ran
    straight afterwards would fail a release that is fine."""
    attempts = []
    waited = []

    def fake_check_call(command, **_):
        attempts.append(command)
        if len(attempts) < 3:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(published, "check_call", fake_check_call)
    published.wait_for_release(
        tmp_path / "served", attempts=6, delay=0.0, sleep=waited.append
    )

    assert len(waited) == 2


def test_a_release_the_index_never_serves_fails_with_what_pip_said(
    monkeypatch, tmp_path
):
    """Reporting "not published yet" after two minutes of trying would hide
    whatever the failure actually was."""

    def fake_check_call(command, **_):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(published, "check_call", fake_check_call)
    with pytest.raises(subprocess.CalledProcessError):
        published.wait_for_release(
            tmp_path / "served", attempts=2, delay=0.0, sleep=lambda _: None
        )


def test_the_check_requirements_are_installed_beside_the_release(monkeypatch, tmp_path):
    """The stages import `packaging`, and the wheel is not required to bring
    it — the same requirement `wheeltest.environment` adds to its venvs."""
    commands = []

    monkeypatch.setattr(
        published, "check_call", lambda command, **_: commands.append(command)
    )
    published.install(tmp_path / "venv", ["dolfinx-solver==1.0"])

    assert "packaging" in commands[-1]


def test_the_second_variant_goes_on_top_of_a_working_install(monkeypatch, tmp_path):
    """Ticket 34's case is reached by installing one variant and then the
    other over it, which is a different transaction from `pip install A B`
    and the one a user actually performs."""
    commands = []

    monkeypatch.setattr(
        published, "check_call", lambda command, **_: commands.append(command)
    )
    monkeypatch.setattr(published, "check_meta", lambda *_: None)
    monkeypatch.setattr(published, "check_both_variants", lambda _: None)
    published.check(tmp_path, version="1.0")

    installs = [command for command in commands if "install" in command]
    for name in published.VARIANT_DISTRIBUTIONS:
        assert sum(f"{name}==1.0" in command for command in installs) == 1
