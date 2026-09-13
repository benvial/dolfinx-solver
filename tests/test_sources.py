"""What has to be true of a source archive before anything unpacks it.

Every vendored binary in the wheel starts as a tarball fetched over HTTPS,
and until ticket 10 nothing checked that the bytes which arrived were the
ones the version pin stood for. These are the rules of that check, and the
rule that every driver naming a URL also names a digest.
"""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from wheelbuild import adios2, dolfinx, kahip, mpich, petsc, slepc, sources

#: Every driver that fetches a source archive, with the constant it records
#: the digest in. A driver added here without a digest fails the sweep below
#: rather than shipping a stage nothing verifies.
DRIVERS = [
    pytest.param(mpich, "MPICH_SHA256", id="mpich"),
    pytest.param(petsc, "PETSC_SHA256", id="petsc"),
    pytest.param(slepc, "SLEPC_SHA256", id="slepc"),
    pytest.param(adios2, "ADIOS2_SHA256", id="adios2"),
    pytest.param(kahip, "KAHIP_SHA256", id="kahip"),
    pytest.param(dolfinx, "DOLFINX_SHA256", id="dolfinx"),
]

PINNED = b"the bytes this build was pinned against"
PINNED_SHA256 = hashlib.sha256(PINNED).hexdigest()


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "source.tar.gz"
    path.write_bytes(PINNED)
    return path


def write_pinned(_url, path):
    """A ``download`` that serves what this build pinned."""
    path.write_bytes(PINNED)
    return path


def counting(calls, payload):
    """A ``download`` that serves ``payload`` and records that it was asked."""

    def download(_url, path):
        calls.append(path)
        path.write_bytes(payload)
        return path

    return download


def test_the_digest_is_the_one_sha256sum_prints(archive):
    assert sources.digest(archive) == PINNED_SHA256


def test_a_digest_is_read_in_blocks_rather_than_whole(tmp_path):
    """The largest archive is 37 MB, and the answer cannot depend on that."""
    big = tmp_path / "big.tar.gz"
    big.write_bytes(b"x" * (3 * sources.BLOCK_SIZE + 17))

    assert sources.digest(big, block_size=64) == sources.digest(big)


def test_matching_bytes_are_no_problem(archive):
    assert sources.digest_problem(PINNED_SHA256, PINNED_SHA256, archive=archive) is None


def test_a_mismatch_names_both_digests(archive):
    problem = sources.digest_problem("0" * 64, PINNED_SHA256, archive=archive)

    assert problem is not None
    assert "0" * 64 in problem
    assert PINNED_SHA256 in problem
    assert str(archive) in problem


def test_a_mismatch_names_where_the_bytes_came_from(archive):
    problem = sources.digest_problem(
        "0" * 64, PINNED_SHA256, archive=archive, url="https://example.invalid/x.tar.gz"
    )

    assert problem is not None
    assert "https://example.invalid/x.tar.gz" in problem


def test_a_missing_digest_is_reported_as_missing_not_as_a_mismatch(archive):
    """A driver that names a URL and no digest is the case worth naming."""
    problem = sources.digest_problem(PINNED_SHA256, "", archive=archive)

    assert problem is not None
    assert "no SHA-256 was recorded" in problem


@pytest.mark.parametrize(
    "expected",
    ["abc", PINNED_SHA256.upper(), PINNED_SHA256 + "0", "z" * 64],
    ids=["truncated", "uppercase", "too-long", "not-hex"],
)
def test_a_digest_that_is_not_a_sha256_is_reported_as_such(expected):
    problem = sources.expected_problem(expected)

    assert problem is not None
    assert "is not a SHA-256" in problem


def test_verify_returns_the_digest_when_the_bytes_are_the_pinned_ones(archive):
    assert sources.verify(archive, PINNED_SHA256) == PINNED_SHA256


def test_verify_refuses_bytes_that_are_not_the_pinned_ones(archive):
    with pytest.raises(ValueError, match="is not the archive this build pinned"):
        sources.verify(archive, "0" * 64)


def test_a_refused_archive_is_left_where_it_is(archive):
    """A retry that happened to succeed would hide the one signal there is."""
    with pytest.raises(ValueError):
        sources.verify(archive, "0" * 64)

    assert archive.exists()


def test_a_cached_archive_that_is_the_pinned_one_is_not_downloaded_again(
    archive, monkeypatch
):
    """A warm cache is the whole reason the archive is kept between runs."""

    def never(*_args, **_kwargs):
        raise AssertionError("the pinned archive must not be re-downloaded")

    monkeypatch.setattr(sources, "download", never)

    assert sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)


def test_a_cached_archive_is_hashed_rather_than_trusted_for_being_there(
    archive, monkeypatch
):
    """Presence under the right name says nothing: the name is the release.

    This is the stuck cache of ticket 26 at the unit level. A tarball
    re-rolled upstream lands on the file the previous one left behind, so
    refusing what is found there is a failure no later run can clear -- the
    bytes on disk would never change.
    """
    monkeypatch.setattr(sources, "download", write_pinned)
    archive.write_bytes(b"tampered with after the download that fetched it")

    sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)

    assert archive.read_bytes() == PINNED


def test_bytes_this_run_downloaded_are_judged_once_and_not_fetched_again(
    tmp_path, monkeypatch
):
    """A retry that happened to succeed would make the one signal transient."""
    downloads = []

    def serve_something_else(_url, path):
        downloads.append(path)
        path.write_bytes(b"not what this build pinned")
        return path

    monkeypatch.setattr(sources, "download", serve_something_else)

    with pytest.raises(ValueError, match="is not the archive this build pinned"):
        sources.fetch(
            "https://example.invalid/x.tar.gz", tmp_path / "x.tar.gz", PINNED_SHA256
        )

    assert len(downloads) == 1


def test_a_refused_download_is_not_fetched_again_on_the_next_run(tmp_path, monkeypatch):
    """Ticket 10's rule, kept through ticket 26's fix.

    A refused download is left where it is, so on the next run it looks
    exactly like a stale cache entry -- same path, same wrong digest. Without
    the marker beside it, re-running a failed job would ask upstream again and
    could pass on the second attempt, which is the transient ticket 10 refused
    to allow.
    """
    archive = tmp_path / "source.tar.gz"
    downloads = []
    monkeypatch.setattr(
        sources, "download", counting(downloads, b"not the pinned bytes")
    )

    with pytest.raises(ValueError, match="is not the archive this build pinned"):
        sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)
    with pytest.raises(ValueError, match="downloaded and refused on an earlier run"):
        sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)

    assert len(downloads) == 1


def test_a_standing_refusal_names_the_archive_and_how_to_clear_it(
    tmp_path, monkeypatch
):
    """A refusal nobody can act on is a stuck cache with better wording."""
    archive = tmp_path / "source.tar.gz"
    monkeypatch.setattr(sources, "download", counting([], b"not the pinned bytes"))

    with pytest.raises(ValueError):
        sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)
    with pytest.raises(ValueError) as refused:
        sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)

    assert str(archive) in str(refused.value)
    assert "delete the archive" in str(refused.value)


def test_a_refusal_stops_standing_when_the_recorded_digest_moves(tmp_path, monkeypatch):
    """Otherwise the marker would re-strand the cache ticket 26 unstranded."""
    archive = tmp_path / "source.tar.gz"
    monkeypatch.setattr(sources, "download", counting([], b"not the pinned bytes"))
    with pytest.raises(ValueError):
        sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)

    # The driver's digest was corrected against what upstream now serves.
    rerolled = b"the release as upstream re-rolled it"
    downloads = []
    monkeypatch.setattr(sources, "download", counting(downloads, rerolled))

    sources.fetch(
        "https://example.invalid/x.tar.gz",
        archive,
        hashlib.sha256(rerolled).hexdigest(),
    )

    assert downloads, "the bumped digest describes a state the refusal did not"


def test_a_refusal_stops_standing_when_the_archive_is_deleted(tmp_path, monkeypatch):
    """The escape hatch the message names has to be the one that works."""
    archive = tmp_path / "source.tar.gz"
    monkeypatch.setattr(sources, "download", counting([], b"not the pinned bytes"))
    with pytest.raises(ValueError):
        sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)

    archive.unlink()
    monkeypatch.setattr(sources, "download", write_pinned)

    assert sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)


def test_a_fetch_that_succeeds_leaves_no_refusal_behind(tmp_path, monkeypatch):
    """A marker outliving what it describes would refuse a good archive."""
    archive = tmp_path / "source.tar.gz"
    monkeypatch.setattr(sources, "download", counting([], b"not the pinned bytes"))
    with pytest.raises(ValueError):
        sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)

    archive.unlink()
    monkeypatch.setattr(sources, "download", write_pinned)
    sources.fetch("https://example.invalid/x.tar.gz", archive, PINNED_SHA256)

    assert not sources.rejection(archive).exists()


def test_a_malformed_recorded_digest_is_reported_without_a_download(
    tmp_path, monkeypatch
):
    """Nothing upstream serves can satisfy a digest that is not one."""

    def never(*_args, **_kwargs):
        raise AssertionError("a typo in the driver is not fetched against")

    monkeypatch.setattr(sources, "download", never)

    with pytest.raises(ValueError, match="is not a SHA-256"):
        sources.fetch("https://example.invalid/x.tar.gz", tmp_path / "x.tar.gz", "abc")


def test_a_missing_archive_is_downloaded_and_then_verified(tmp_path, monkeypatch):
    target = tmp_path / "source.tar.gz"
    monkeypatch.setattr(sources, "download", write_pinned)

    assert sources.fetch("https://example.invalid/x.tar.gz", target, PINNED_SHA256)


def test_nothing_is_fetched_over_plain_http(tmp_path):
    with pytest.raises(ValueError, match="is not an https URL"):
        sources.download("http://example.invalid/x.tar.gz", tmp_path / "x.tar.gz")


def test_a_redirect_off_https_is_refused():
    handler = sources.HttpsOnlyRedirects()

    with pytest.raises(ValueError, match="refusing to follow a redirect"):
        handler.redirect_request(None, None, 302, "Found", None, "http://elsewhere/x")


@pytest.mark.parametrize(("driver", "constant"), DRIVERS)
def test_every_driver_that_names_a_url_names_a_digest(driver, constant):
    recorded = getattr(driver, constant)

    assert sources.expected_problem(recorded) is None


@pytest.mark.parametrize(("driver", "constant"), DRIVERS)
def test_the_digest_is_recorded_beside_the_version_it_pins(driver, constant):
    """One place to move when a version moves, which is the whole point."""
    source = Path(driver.__file__)
    version_constant = constant.replace("_SHA256", "_VERSION")
    lines = [
        number
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines())
        if line.startswith((constant, version_constant))
    ]

    assert len(lines) == 2, f"{constant} and its version are not both in {source}"
    assert lines[1] - lines[0] < 25


def test_no_two_archives_are_pinned_to_the_same_digest():
    """Two identical values would be a copy-paste, not two identical files."""
    recorded = [getattr(param.values[0], param.values[1]) for param in DRIVERS]

    assert len(set(recorded)) == len(recorded)


def test_the_cli_passes_an_archive_that_hashes_to_the_recorded_digest(archive, capsys):
    assert sources.main(["--archive", str(archive), "--expected", PINNED_SHA256]) == 0
    assert PINNED_SHA256 in capsys.readouterr().out


def test_the_cli_fails_on_an_archive_that_does_not(archive, capsys):
    assert sources.main(["--archive", str(archive), "--expected", "0" * 64]) == 1
    assert "is not the archive this build pinned" in capsys.readouterr().err


def test_the_cli_fails_on_an_archive_that_is_not_there(tmp_path, capsys):
    missing = tmp_path / "never-downloaded.tar.gz"

    assert sources.main(["--archive", str(missing), "--expected", "0" * 64]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_the_dolfinx_digest_is_the_one_the_demo_stage_uses(tmp_path, monkeypatch):
    """One recorded value covers the build's stage and the demo stage both."""
    from wheeltest import demos

    cache = tmp_path / "demo-source"
    cache.mkdir()

    def write_something_else(_url, path):
        with tarfile.open(path, "w:gz"):
            pass
        return path

    monkeypatch.setattr(sources, "download", write_something_else)

    with pytest.raises(ValueError, match=dolfinx.DOLFINX_SHA256):
        demos.fetch(cache)


def demo_tarball(path, *, text=b"a demo program"):
    """An archive laid out the way upstream's release tarball is."""
    from wheeltest import demos

    with tarfile.open(path, "w:gz") as tar:
        member = tarfile.TarInfo(
            f"{dolfinx.source_dir_name()}/{demos.DEMO_RELATIVE}/{demos.DEMOS[0].name}"
        )
        member.size = len(text)
        tar.addfile(member, io.BytesIO(text))
    return path


def test_the_demo_cache_is_re_fetched_when_the_digest_moves(tmp_path):
    """A version that did not move cannot be what decides this (ticket 10).

    The cache holds an *extracted* tree named after the release, and nothing
    re-hashes a tree — so without a marker a re-rolled tarball would keep
    being run out of the bytes it replaced.
    """
    from wheeltest import demos

    cache = tmp_path / "demo-source"
    archive = cache / f"{dolfinx.source_dir_name()}.tar.gz"
    cache.mkdir()
    demo_tarball(archive, text=b"the demo as it was")
    demos.fetch(cache, expected=sources.digest(archive))

    demo_tarball(archive, text=b"the demo as it is now")
    rerolled = sources.digest(archive)
    directory = demos.fetch(cache, expected=rerolled)

    assert (directory / demos.DEMOS[0].name).read_bytes() == b"the demo as it is now"
    marker = cache / dolfinx.source_dir_name() / demos.EXTRACTED_MARKER
    assert marker.read_text().strip() == rerolled


def test_a_digest_bump_recovers_a_cache_holding_the_previous_archive(
    tmp_path, monkeypatch
):
    """The case ticket 26 found: no manual deletion, one run, no stranding.

    Every warm cache holds the previous tarball under the name the new one
    wants, because the file is named for the release and a re-rolled release
    keeps its number. The stale archive is deliberately left on disk here --
    rewriting it first is the step the field never performs.
    """
    from wheeltest import demos

    cache = tmp_path / "demo-source"
    stale = cache / f"{dolfinx.source_dir_name()}.tar.gz"
    cache.mkdir()
    demo_tarball(stale, text=b"the demo as it was")
    demos.fetch(cache, expected=sources.digest(stale))

    rerolled = demo_tarball(tmp_path / "rerolled.tar.gz", text=b"the demo as it is now")
    expected = sources.digest(rerolled)
    monkeypatch.setattr(
        sources, "download", lambda _url, path: path.write_bytes(rerolled.read_bytes())
    )

    directory = demos.fetch(cache, expected=expected)

    assert (directory / demos.DEMOS[0].name).read_bytes() == b"the demo as it is now"
    marker = cache / dolfinx.source_dir_name() / demos.EXTRACTED_MARKER
    assert marker.read_text().strip() == expected


def test_a_re_fetch_that_fails_leaves_the_tree_it_could_not_replace(
    tmp_path, monkeypatch
):
    """Destroying the only good copy first is what made the failure permanent."""
    from wheeltest import demos

    cache = tmp_path / "demo-source"
    archive = cache / f"{dolfinx.source_dir_name()}.tar.gz"
    cache.mkdir()
    demo_tarball(archive, text=b"the demo as it was")
    demos.fetch(cache, expected=sources.digest(archive))

    def serve_something_else(_url, path):
        demo_tarball(path, text=b"something nobody pinned")
        return path

    monkeypatch.setattr(sources, "download", serve_something_else)

    with pytest.raises(ValueError, match="is not the archive this build pinned"):
        demos.fetch(cache, expected="0" * 64)

    demo = cache / dolfinx.source_dir_name() / demos.DEMO_RELATIVE / demos.DEMOS[0].name
    assert demo.read_bytes() == b"the demo as it was"


def test_an_unchanged_demo_cache_is_reused(tmp_path):
    """The marker must not turn a warm cache into a download every run."""
    from wheeltest import demos

    cache = tmp_path / "demo-source"
    archive = cache / f"{dolfinx.source_dir_name()}.tar.gz"
    cache.mkdir()
    demo_tarball(archive)
    expected = sources.digest(archive)
    demos.fetch(cache, expected=expected)

    archive.unlink()

    assert demos.fetch(cache, expected=expected).is_dir()
