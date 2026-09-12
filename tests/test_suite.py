"""Which stages the suite runs, and how it invokes each of them."""

from pathlib import Path

import pytest

from wheeltest import suite


def _command(stage: suite.Stage, **overrides) -> list[str]:
    arguments = {
        "python": Path("/venv/bin/python"),
        "site": Path("/venv/lib/python3.12/site-packages"),
        "launcher": Path("/venv/bin/mpiexec"),
        "work_dir": Path("/work"),
    }
    arguments.update(overrides)
    return suite.command(stage, **arguments)


def test_the_import_stage_is_the_builds_own_check():
    """Written twice, the two definitions of a working import would drift."""
    assert suite.STAGES[0].module == "wheelbuild.import_check"


def test_the_import_stage_asks_for_the_dolfinx_half():
    command = _command(suite.stage_by_name("imports"))

    assert command[:3] == ["/venv/bin/python", "-m", "wheelbuild.import_check"]
    assert "--dolfinx" in command
    assert "--site" in command


def test_a_serial_stage_runs_in_the_venvs_interpreter():
    command = _command(suite.stage_by_name("smoke"))

    assert command[0] == "/venv/bin/python"
    assert "mpiexec" not in " ".join(command)


def test_the_interop_stage_runs_under_two_ranks():
    command = _command(suite.stage_by_name("interop"))

    assert command[:3] == ["/venv/bin/mpiexec", "-n", "2"]
    assert "wheeltest.interop" in command


def test_the_demo_stage_is_told_where_the_sources_are():
    command = _command(
        suite.stage_by_name("demos"), demo_source=Path("/build/dolfinx-0.11.0.post0")
    )

    assert "--source" in command
    assert "/build/dolfinx-0.11.0.post0" in command
    assert "--cache" not in command


def test_the_demo_stage_downloads_when_no_source_tree_is_given():
    command = _command(suite.stage_by_name("demos"), demo_cache=Path("/cache"))

    assert command[command.index("--cache") + 1] == "/cache"


def test_only_the_demo_stage_takes_demo_arguments():
    for stage in suite.STAGES:
        if stage.name == suite.DEMOS_STAGE:
            continue
        command = _command(
            stage, demo_source=Path("/build/dolfinx"), demo_cache=Path("/c")
        )
        assert "--source" not in command
        assert "--work-dir" not in command


def test_every_stage_is_told_which_install_it_is_asking():
    """Three ways to fail, and the message has to say which install it was."""
    for stage in suite.STAGES:
        command = _command(stage)
        assert (
            command[command.index("--site") + 1] == "/venv/lib/python3.12/site-packages"
        )


def test_every_stage_says_what_a_pass_means():
    for stage in suite.STAGES:
        assert stage.proves
        assert stage.ranks >= 1


def test_a_stage_can_be_named_to_re_run_it_alone():
    assert suite.stage_by_name("demos").module == "wheeltest.demos"


def test_an_unknown_stage_lists_the_ones_there_are():
    with pytest.raises(ValueError, match="imports, foreign, smoke, interop, demos"):
        suite.stage_by_name("numerics")
