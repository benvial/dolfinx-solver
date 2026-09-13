"""What a tag is allowed to upload, checked before the upload happens.

PyPI never reuses a filename, so the gathering step is the last place a wrong
file can be stopped. These are the cases it has to stop: a wheel out of the
build's own wheelhouse, a half release, and a file left behind by another
run.
"""

from pathlib import Path

import pytest

from wheelbuild import publish
from wheelbuild.petsc import SCALAR_TYPES
from wheelbuild.version import __version__

#: A release, as file names spell it.
COMPLEX_WHEEL = (
    f"dolfinx_solver_complex-{__version__}-cp312-abi3-manylinux_2_34_x86_64.whl"
)
REAL_WHEEL = f"dolfinx_solver_real-{__version__}-cp312-abi3-manylinux_2_34_x86_64.whl"
META_WHEEL = f"dolfinx_solver-{__version__}-py3-none-any.whl"
META_SDIST = f"dolfinx_solver-{__version__}.tar.gz"

RELEASE = (COMPLEX_WHEEL, REAL_WHEEL, META_WHEEL, META_SDIST)


def _paths(*names: str) -> list[Path]:
    return [Path("dist") / name for name in names]


def test_a_release_is_the_two_variants_and_the_meta_package():
    assert publish.gathered_problem(_paths(*RELEASE)) is None


def test_the_variants_are_every_one_the_drivers_build():
    """The declared variant is what every other driver runs for; a release is
    all of them (ticket 21)."""
    expected = tuple(f"dolfinx-solver-{variant}" for variant in SCALAR_TYPES)

    assert expected == publish.VARIANT_DISTRIBUTIONS
    assert publish.META_DISTRIBUTION not in publish.VARIANT_DISTRIBUTIONS


def test_the_meta_package_is_published_beside_the_variants_it_pins():
    assert set(publish.DISTRIBUTIONS) == {
        *publish.VARIANT_DISTRIBUTIONS,
        publish.META_DISTRIBUTION,
    }


@pytest.mark.parametrize("missing", [COMPLEX_WHEEL, REAL_WHEEL, META_WHEEL])
def test_a_release_missing_a_wheel_is_refused(missing):
    problem = publish.gathered_problem(
        _paths(*[name for name in RELEASE if name != missing])
    )

    assert problem is not None
    assert "release" in problem


def test_the_meta_packages_sdist_is_not_required():
    """`python -m build` makes both; only the wheel is what pip installs."""
    assert (
        publish.gathered_problem(
            _paths(*[name for name in RELEASE if name != META_SDIST])
        )
        is None
    )


def test_a_source_distribution_of_a_variant_is_refused():
    """An sdist of a variant on the index is an install that tries to compile
    PETSc, SLEPc, ADIOS2 and DOLFINx on the user's machine."""
    problem = publish.gathered_problem(
        _paths(*RELEASE, f"dolfinx_solver_complex-{__version__}.tar.gz")
    )

    assert problem is not None
    assert "source distribution" in problem


def test_a_wheel_this_project_does_not_publish_is_refused():
    """The build's other wheelhouse holds petsc4py, slepc4py and
    fenics_dolfinx wheels built for the build's own use (spec §6)."""
    problem = publish.gathered_problem(
        _paths(*RELEASE, "petsc4py-3.24.0-cp312-cp312-linux_x86_64.whl")
    )

    assert problem is not None
    assert "petsc4py" in problem


@pytest.mark.parametrize("stray", ["README.txt", "notes"])
def test_a_file_that_is_no_distribution_at_all_is_refused(stray):
    """Whatever is in the directory is what gets uploaded."""
    problem = publish.gathered_problem(_paths(*RELEASE, stray))

    assert problem is not None
    assert stray in problem


def test_a_file_from_another_release_is_refused():
    problem = publish.gathered_problem(
        _paths(*RELEASE, "dolfinx_solver-0.10.0-py3-none-any.whl")
    )

    assert problem is not None
    assert "0.10.0" in problem


def test_two_wheels_of_one_variant_are_refused():
    """A stale artefact beside a fresh one; which PyPI would serve is not
    something to guess at."""
    problem = publish.gathered_problem(
        _paths(
            *RELEASE,
            f"dolfinx_solver_complex-{__version__}-cp312-abi3-manylinux_2_28_x86_64.whl",
        )
    )

    assert problem is not None
    assert "dolfinx-solver-complex" in problem


def test_the_version_a_release_is_gathered_for_is_the_packaged_one():
    """The tag check tied the tag to it before the build started."""
    problem = publish.gathered_problem(_paths(*RELEASE), version="9.9.9")

    assert problem is not None
    assert "9.9.9" in problem


def test_both_indexes_are_named_from_both_ends():
    """Spec §9: one manual TestPyPI publish before the first real release.
    Upload and install are different services and neither URL follows from
    the other, so an index is the pair."""
    assert set(publish.INDEXES) == {"pypi", "testpypi"}
    assert "test.pypi.org" in publish.INDEXES["testpypi"].upload
    assert "test.pypi.org" in str(publish.INDEXES["testpypi"].simple)
    assert "test" not in publish.INDEXES["pypi"].upload


def test_the_real_index_is_installed_from_the_way_a_user_installs():
    """pip's own default, with no argument steering it: the dry run is the
    one that needs telling where to look."""
    assert publish.INDEXES["pypi"].simple is None


def test_what_pip_install_dolfinx_solver_means_is_read_from_the_pin():
    """Not derived from the default scalar type: the decision is that one
    dependency line, and a release path asserting the derived answer would
    agree with itself while disagreeing with what it published (spec §1)."""
    assert publish.pinned_distribution() in publish.VARIANT_DISTRIBUTIONS


def test_a_meta_package_that_pins_more_than_one_thing_is_refused(tmp_path):
    """It is a name and a pin; anything else is a decision nobody recorded."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\ndependencies = ["dolfinx-solver-complex==1.0", "numpy"]\n'
    )

    with pytest.raises(ValueError, match="one pin"):
        publish.pinned_distribution(pyproject)


def test_everything_in_the_wheelhouse_is_collected_not_only_what_looks_right(tmp_path):
    """A filter here would hide the artefact the check exists to catch."""
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / COMPLEX_WHEEL).write_bytes(b"")
    (wheelhouse / "petsc4py-3.24.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"")

    collected = publish.collect_wheels(wheelhouse, tmp_path / "dist")

    assert {path.name for path in collected} == {
        COMPLEX_WHEEL,
        "petsc4py-3.24.0-cp312-cp312-linux_x86_64.whl",
    }


def test_a_wheelhouse_that_was_never_downloaded_is_not_an_empty_release(tmp_path):
    with pytest.raises(FileNotFoundError, match="wheelhouse"):
        publish.collect_wheels(tmp_path / "missing", tmp_path / "dist")


def test_the_same_version_spelled_two_ways_is_one_release():
    """File names are written by the tool that built them, always
    normalised; `__version__` is written by hand. Which spelling a tag uses
    is `wheelbuild.tag_check`'s subject, not this one's."""
    assert publish.gathered_problem(_paths(*RELEASE), version=f"{__version__}0") is None
