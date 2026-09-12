"""The demo subset: what it has to cover, and how it is run."""

import pytest

from wheelbuild import dolfinx as dolfinx_driver
from wheelbuild import petsc
from wheeltest import demos


@pytest.mark.parametrize("variant", ["complex", "real"])
def test_the_subset_is_the_size_the_spec_asks_for(variant):
    """Spec §11: three to five upstream demos."""
    assert 3 <= len(demos.subset(variant)) <= 5


@pytest.mark.parametrize("variant", ["complex", "real"])
def test_every_subset_holds_a_parallel_demo(variant):
    assert any(demo.ranks > 1 for demo in demos.subset(variant))


def test_the_complex_subset_holds_the_demos_written_for_complex_scalars():
    assert demos.HELMHOLTZ in demos.DEMO_SUBSETS["complex"]
    assert demos.WAVEGUIDE in demos.DEMO_SUBSETS["complex"]


def test_the_real_subset_holds_none_of_them():
    """They interpolate `exp(1j...)`: under a real build numpy drops the
    imaginary part, the demo exits zero, and the stage proves nothing."""
    assert demos.HELMHOLTZ not in demos.DEMO_SUBSETS["real"]
    assert demos.WAVEGUIDE not in demos.DEMO_SUBSETS["real"]


def test_every_variant_the_build_knows_has_a_subset():
    assert set(demos.DEMO_SUBSETS) == set(petsc.SCALAR_TYPES)


def test_a_variant_this_build_has_no_name_for_is_refused():
    with pytest.raises(ValueError, match="Complex"):
        demos.subset("Complex")


def test_the_subset_run_is_the_one_the_build_declares():
    assert demos.DEMO_SUBSETS[petsc.SCALAR_TYPE] == demos.DEMOS


def test_the_subsets_agree_on_what_the_first_demo_is():
    """`demo_dir` recognises a source tree by it, for either variant."""
    assert {subset[0] for subset in demos.DEMO_SUBSETS.values()} == {demos.POISSON}


@pytest.mark.parametrize("variant", ["complex", "real"])
def test_every_demo_says_what_it_covers(variant):
    for demo in demos.subset(variant):
        assert demo.covers
        assert demo.name.startswith("demo_")
        assert demo.ranks >= 1


def test_a_serial_demo_runs_as_itself(tmp_path):
    command = demos.command(
        demos.Demo("demo_helmholtz.py", 1, "complex"),
        tmp_path,
        tmp_path / "bin" / "python",
        tmp_path / "bin" / "mpiexec",
    )

    assert command == [
        str(tmp_path / "bin" / "python"),
        str(tmp_path / "demo_helmholtz.py"),
    ]


def test_a_parallel_demo_runs_under_the_wheels_own_launcher(tmp_path):
    command = demos.command(
        demos.Demo("demo_poisson.py", 2, "parallel"),
        tmp_path,
        tmp_path / "bin" / "python",
        tmp_path / "bin" / "mpiexec",
    )

    assert command[:3] == [str(tmp_path / "bin" / "mpiexec"), "-n", "2"]


def test_a_demo_that_ran_passes():
    assert demos.outcome_problem(demos.DEMOS[0], 0, "") is None


def test_a_demo_that_failed_says_what_it_was_covering():
    problem = demos.outcome_problem(demos.DEMOS[0], 1, "Traceback\nRuntimeError: no")

    assert problem is not None
    assert demos.DEMOS[0].covers[:20] in problem
    assert "RuntimeError: no" in problem


def test_a_demo_reading_the_version_from_metadata_is_a_packaging_failure():
    """The one thing in the subset that fails for a packaging reason (ticket 14)."""
    problem = demos.metadata_problem([("demo_x.py", 'v = version("fenics-dolfinx")\n')])

    assert problem is not None
    assert "PackageNotFoundError" in problem


def test_demos_that_ask_dolfinx_itself_are_fine():
    assert (
        demos.metadata_problem([("demo_x.py", "print(dolfinx.__version__)\n")]) is None
    )


def test_the_demo_directory_is_found_inside_a_source_tree(tmp_path):
    inside = tmp_path / demos.DEMO_RELATIVE
    inside.mkdir(parents=True)

    assert demos.demo_dir(tmp_path) == inside


def test_the_demo_directory_is_accepted_directly(tmp_path):
    (tmp_path / demos.DEMOS[0].name).write_text("")

    assert demos.demo_dir(tmp_path) == tmp_path


def test_some_other_directory_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither a DOLFINx source tree"):
        demos.demo_dir(tmp_path)


def test_a_renamed_upstream_demo_is_refused_rather_than_skipped(tmp_path):
    """A subset that quietly shrinks is a test suite that quietly stops testing."""
    for demo in demos.DEMOS[1:]:
        (tmp_path / demo.name).write_text("")

    with pytest.raises(FileNotFoundError, match=demos.DEMOS[0].name):
        demos.check(
            tmp_path, tmp_path / "work", tmp_path / "python", tmp_path / "mpiexec"
        )


def test_the_demos_come_from_the_release_this_wheel_mirrors(tmp_path):
    """A newer release's demos would be testing upstream's drift, not the wheel."""
    with pytest.raises(ValueError, match="not an https URL"):
        demos.fetch(tmp_path, url="http://example.invalid/dolfinx.tar.gz")

    assert dolfinx_driver.DOLFINX_VERSION in dolfinx_driver.source_url()


def test_a_demo_runs_the_way_a_user_runs_it():
    """Upstream code knows nothing of this repository, so it is off the path."""
    child = demos.child_environment({"PYTHONPATH": "/repo", "HOME": "/home/user"})

    assert "PYTHONPATH" not in child
    assert child["HOME"] == "/home/user"
