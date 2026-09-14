"""What an unattended weekly bump is allowed to do, and where it stops.

The job this covers runs with nobody watching and opens a pull request against
this repository. Two failures matter more than the rest: a substitution that
places nothing — a pull request that looks like a bump and builds the old
release — and a bump of something that was a decision rather than a copy. The
tests below are mostly the first: every rewrite is applied to the file that is
actually checked in, so a bump driver that has drifted from the files it edits
fails here rather than at 06:17 on a Monday.
"""

import json
import re
import tomllib

import pytest

from wheelbuild import dolfinx, mpich, release_check, sources
from wheelbuild.version import DOLFINX_VERSION

#: A release newer than anything this repository has mirrored, standing in for
#: whatever upstream tags next.
NEXT = "0.12.0.post1"

#: Upstream's own metadata, cut down to what the bump reads out of it.
UPSTREAM_PYPROJECT = """\
[build-system]
requires = ["scikit-build-core[pyproject]>=0.11", "nanobind>=2.5.0", "mpi4py"]

[project]
name = "fenics-dolfinx"
version = "0.12.0.post1"
dependencies = [
      "numpy>=2",
      "fenics-basix>=0.12.0,<0.13.0",
      "fenics-ffcx>=0.12.0,<0.13.0",
      "fenics-ufl>=2026.2.0,<2026.3.0",
]
"""


def checked_in(path):
    """Read one of the files the bump rewrites, as it is committed."""
    return path.read_text(encoding="utf-8")


def test_the_release_is_the_version_without_upstreams_packaging_segment():
    assert release_check.release_of("0.12.0.post1") == "0.12.0"
    assert release_check.release_of("0.12.0") == "0.12.0"


@pytest.mark.parametrize("version", ["0.12.0", "0.12.0.post3"])
def test_an_ordinary_release_is_mirrorable(version):
    assert release_check.mirrorable_problem(version) is None


@pytest.mark.parametrize("version", ["0.12.0rc1", "0.12.0.dev1", "0.12.0+local"])
def test_a_release_this_packaging_cannot_number_is_refused(version):
    """`RELEASE` is the version without a `.postN` and the tests assert that
    nothing else sits between the two, so an rc is a bump for a human."""
    problem = release_check.mirrorable_problem(version)

    assert problem is not None
    assert version in problem


def test_a_newer_upstream_release_is_ahead():
    assert release_check.is_behind(DOLFINX_VERSION, NEXT)


def test_a_release_upstream_took_backwards_is_not_news():
    """A yanked release moves `latest` down, and a string comparison would
    open a pull request undoing the version this repository ships."""
    assert not release_check.is_behind("0.11.0.post2", "0.11.0.post1")


def test_packaging_segments_are_compared_as_numbers():
    assert release_check.is_behind("0.11.0.post9", "0.11.0.post10")


def test_the_release_the_repository_already_ships_is_not_a_bump():
    assert not release_check.is_behind(DOLFINX_VERSION, DOLFINX_VERSION)


def test_the_latest_release_is_read_off_the_tag(tmp_path, monkeypatch):
    def answer(_url, archive):
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(json.dumps({"tag_name": f"v{NEXT}"}), encoding="utf-8")
        return archive

    monkeypatch.setattr(release_check.sources, "download", answer)

    assert release_check.latest_upstream_release(tmp_path) == NEXT


def test_an_answer_that_names_no_release_is_not_a_pass(tmp_path, monkeypatch):
    """Deciding "not behind" because GitHub could not be read is how a weekly
    job stops working without anyone noticing."""

    def answer(_url, archive):
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text("{}", encoding="utf-8")
        return archive

    monkeypatch.setattr(release_check.sources, "download", answer)

    with pytest.raises(OSError, match="tag_name"):
        release_check.latest_upstream_release(tmp_path)


def test_a_substitution_that_places_nothing_is_refused():
    with pytest.raises(ValueError, match="drifted apart"):
        release_check.substitute(
            "nothing here\n", re.compile("^absent$", re.MULTILINE), "x", where="f"
        )


def test_a_substitution_with_two_candidates_is_refused():
    with pytest.raises(ValueError, match="has 2 lines"):
        release_check.substitute(
            "line\nline\n", re.compile("^line$", re.MULTILINE), "x", where="f"
        )


def test_a_line_that_already_says_it_is_no_change():
    text, change = release_check.substitute(
        "value = 1\n", re.compile(r"^value = 1$", re.MULTILINE), "value = 1", where="f"
    )

    assert text == "value = 1\n"
    assert change is None


def test_the_version_module_carries_the_release_in_three_places():
    """`RELEASE`, `DOLFINX_VERSION` and `__version__` are one source with
    three spellings, and a bump that moved two of them would ship a wheel
    numbered for a release it does not build."""
    bumped, changes = release_check.bump_version_module(
        checked_in(release_check.VERSION_MODULE), dolfinx_version=NEXT
    )

    assert f'RELEASE = "{release_check.release_of(NEXT)}"' in bumped
    assert f'DOLFINX_VERSION = "{NEXT}"' in bumped
    assert f'__version__ = "{NEXT}"' in bumped
    assert len(changes) == 3


def test_the_version_module_keeps_its_prose():
    """The literals are rewritten, not the file: the comments around them are
    what tell the next reader why there are three."""
    source = checked_in(release_check.VERSION_MODULE)

    bumped, _ = release_check.bump_version_module(source, dolfinx_version=NEXT)

    assert bumped.count("\n") == source.count("\n")
    assert "the single source of the version" in bumped.lower()


def test_the_source_digest_is_rewritten_where_the_driver_records_it():
    digest = "a" * 64

    bumped, changes = release_check.bump_source_digest(
        checked_in(release_check.DOLFINX_DRIVER), digest=digest
    )

    assert f'DOLFINX_SHA256 = "{digest}"' in bumped
    assert dolfinx.DOLFINX_SHA256 not in bumped
    assert len(changes) == 1


def test_a_digest_that_is_not_one_is_refused_before_anything_is_written():
    """A truncated paste would otherwise be committed and fail every later
    run as a mismatch rather than as the typo it is."""
    with pytest.raises(ValueError):
        release_check.bump_source_digest(
            checked_in(release_check.DOLFINX_DRIVER), digest="abc"
        )


def test_the_trio_pins_are_copied_verbatim_from_the_release():
    bumped, changes = release_check.bump_trio_pins(
        checked_in(release_check.PYPROJECT), UPSTREAM_PYPROJECT
    )

    declared = tomllib.loads(bumped)["project"]["dependencies"]
    for pin in (
        "fenics-basix>=0.12.0,<0.13.0",
        "fenics-ffcx>=0.12.0,<0.13.0",
        "fenics-ufl>=2026.2.0,<2026.3.0",
    ):
        assert pin in declared
    assert len(changes) == 3


def test_only_the_trio_is_touched():
    """`mpich` and `numpy` are this project's own pins; upstream's file has
    opinions about them that are not ours to copy (ADR-0001)."""
    before = tomllib.loads(checked_in(release_check.PYPROJECT))["project"][
        "dependencies"
    ]

    bumped, _ = release_check.bump_trio_pins(
        checked_in(release_check.PYPROJECT), UPSTREAM_PYPROJECT
    )

    after = tomllib.loads(bumped)["project"]["dependencies"]
    untouched = {pin for pin in before if not pin.startswith("fenics-")}
    assert untouched <= set(after)
    assert len(after) == len(before)


def test_a_release_that_declares_none_of_the_trio_stops_the_bump():
    """Upstream restructuring its metadata is exactly when a copy has to stop
    and a human has to re-read the coupling contract (spec §10)."""
    with pytest.raises(ValueError, match="spec §10"):
        release_check.bump_trio_pins(
            checked_in(release_check.PYPROJECT),
            '[project]\nname = "fenics-dolfinx"\ndependencies = ["numpy>=2"]\n',
        )


def test_the_comment_naming_the_mirrored_release_follows_the_pins():
    """It is what tells the next reader which release to re-read them
    against."""
    bumped, changes = release_check.bump_mirrored_release(
        checked_in(release_check.PYPROJECT), dolfinx_version=NEXT
    )

    assert f"at the mirrored release (v{NEXT})" in bumped
    assert f"(v{DOLFINX_VERSION})" not in bumped
    assert len(changes) == 1


def test_the_meta_package_moves_in_lockstep():
    bumped, changes = release_check.bump_meta_project(
        checked_in(release_check.META_PYPROJECT), wheel_version=NEXT
    )

    metadata = tomllib.loads(bumped)["project"]
    assert metadata["version"] == NEXT
    assert metadata["dependencies"] == [f"dolfinx-solver-complex=={NEXT}"]
    assert len(changes) == 2


def test_the_meta_package_keeps_the_variant_it_pins():
    """Which variant `pip install dolfinx-solver` resolves to is a decision
    recorded in that one line, and a version bump is not a place to change
    it."""
    flipped = checked_in(release_check.META_PYPROJECT).replace(
        "dolfinx-solver-complex==", "dolfinx-solver-real=="
    )

    bumped, _ = release_check.bump_meta_project(flipped, wheel_version=NEXT)

    assert f'dependencies = ["dolfinx-solver-real=={NEXT}"]' in bumped


def test_the_nanobind_requirement_is_read_but_never_applied():
    """It is a floor upstream declares; ours is a coupling to the basix
    wheel's ABI, so the report names both and the bump moves neither."""
    assert (
        release_check.upstream_nanobind_requirement(UPSTREAM_PYPROJECT)
        == "nanobind>=2.5.0"
    )


def test_a_release_declaring_no_nanobind_is_reported_as_such():
    assert release_check.upstream_nanobind_requirement("[build-system]\n") is None


def test_the_body_says_what_moved_and_what_did_not():
    changes = [release_check.Bump("pyproject.toml", 'a = "1"', 'a = "2"')]

    body = release_check.report(
        packaged=DOLFINX_VERSION,
        upstream=NEXT,
        changes=changes,
        upstream_pyproject_text=UPSTREAM_PYPROJECT,
    )

    assert f"DOLFINx {NEXT}" in body
    assert DOLFINX_VERSION in body
    assert '`pyproject.toml`: `a = "1"` → `a = "2"`' in body
    assert dolfinx.NANOBIND_VERSION in body
    assert mpich.MPICH_VERSION in body
    assert f"v{NEXT}" in body


def test_the_bump_is_written_into_a_repository_rather_than_a_file(
    tmp_path, monkeypatch
):
    """Every rewrite lands in the copy, and the download is the release's own
    tarball — stubbed here, since the digest is the point of it and a unit
    test may not reach the network."""
    repository = tmp_path / "repo"
    for path in (
        release_check.VERSION_MODULE,
        release_check.DOLFINX_DRIVER,
        release_check.PYPROJECT,
        release_check.META_PYPROJECT,
    ):
        relative = path.relative_to(release_check.PYPROJECT.parent)
        (repository / relative).parent.mkdir(parents=True, exist_ok=True)
        (repository / relative).write_text(checked_in(path), encoding="utf-8")

    def served(_url, archive):
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"tarball")
        return archive

    monkeypatch.setattr(sources, "download", served)
    monkeypatch.setattr(
        release_check,
        "upstream_pyproject",
        lambda _archive, _version: UPSTREAM_PYPROJECT,
    )

    changes, upstream_text = release_check.bump(
        NEXT, work_dir=tmp_path / "downloads", root=repository
    )

    assert upstream_text == UPSTREAM_PYPROJECT
    assert {change.path for change in changes} == {
        "_version.py",
        "dolfinx.py",
        "pyproject.toml",
        "meta/pyproject.toml",
    }
    version_module_text = (repository / "dolfinx_solver" / "_version.py").read_text(
        encoding="utf-8"
    )
    assert f'__version__ = "{NEXT}"' in version_module_text
    digest = sources.digest(tmp_path / "downloads" / f"dolfinx-{NEXT}.tar.gz")
    assert digest in (repository / "wheelbuild" / "dolfinx.py").read_text(
        encoding="utf-8"
    )


def test_a_release_that_cannot_be_mirrored_writes_nothing(tmp_path):
    with pytest.raises(ValueError, match="by hand"):
        release_check.bump("0.12.0rc1", work_dir=tmp_path, root=tmp_path)
