"""Prove a downloaded source archive is the one this build pinned.

Every vendored binary in the published wheel starts as a tarball fetched over
HTTPS from a URL a driver module names. HTTPS says the bytes came from the
host the URL names and arrived unaltered; it says nothing about whether that
host is still serving the same bytes it served when the version was pinned. A
tarball re-rolled upstream under the same name, or a mirror that has drifted,
changes what is inside a published wheel with no signal at all — the version
pin (spec §10) would still read as satisfied.

So every archive is pinned by content as well as by version: each driver
records the SHA-256 of the release it names, next to the version, and nothing
is extracted before the bytes on disk have been shown to hash to it. There is
one recorded value per archive and three things read it — the container
build's ``fetch_source``, :mod:`wheeltest.demos`, and the trio half of
:mod:`wheelbuild.pin_check` — rather than three lists to keep in step.

Bytes that have just been fetched are judged once and never fetched again. A
second download that happened to succeed would turn the one signal this module
exists to raise into a transient, so the archive is left where it is and the
message names it: the question "are these the bytes we pinned?" is answered by
looking at the file, not by trying again.

A *cached* archive is a different question, and ticket 26 is what separated
them. The caches are keyed by release, so a tarball re-rolled upstream lands on
the file the previous one left behind — and refusing it there is a failure no
later run can clear, because nothing ever replaces the bytes being refused.
What such a cache holds is this build's own housekeeping rather than evidence
about upstream, so :func:`fetch` hashes what it finds and downloads over it
when it is not the pinned archive. The postcondition is the function's: after
it returns, the file it names is the archive the driver pinned.

The two rules meet on one file, because a refused download is left where it
is and a stale cache entry looks exactly like it: same path, same wrong
digest. What tells them apart is a *rejection marker* written beside the
archive when a download is refused, naming both the digest that was wanted and
the digest that arrived. While it stands, and while the bytes beside it are
still the ones it describes, the refusal is repeated without fetching
anything — so re-running a failed job reports the same mismatch rather than
quietly passing on the second attempt. It stops standing the moment anything
it describes changes: a recorded digest that moved (the bump ticket 26 is
about), or an archive deleted by the person the message asked to look at it.
It is written and read under :mod:`wheelbuild.markers`' rules, so a run killed
while writing it leaves the previous refusal rather than a file every later run
raises on.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from wheelbuild import markers

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Characters in a hex SHA-256. A digest of any other length is a typo, a
#: truncation or a placeholder, and is reported as such rather than as a
#: mismatch: telling the two apart is what turns "the archive changed" into
#: something worth investigating.
DIGEST_LENGTH = 64

#: The hex alphabet a digest is written in. Lowercase, because that is what
#: ``sha256sum`` prints and what the constants hold, so a pasted uppercase
#: digest is caught here rather than failing every run as a mismatch.
DIGEST_ALPHABET = frozenset("0123456789abcdef")

#: How much of an archive is hashed at a time. The largest of them is MPICH's
#: at 37 MB, which is not worth holding in memory when a loop costs nothing.
BLOCK_SIZE = 1 << 20

#: What a download identifies itself as. GitHub's archive endpoint answers the
#: default ``Python-urllib/3.x`` with a 500 rather than a tarball, so the
#: header is set rather than left to the default — a fetch that fails with
#: "Internal Server Error" against a URL that works in a browser is a long
#: afternoon otherwise.
USER_AGENT = "dolfinx-solver"

#: How long to wait for a source archive. Long enough for MPICH's 37 MB on a
#: slow runner, short enough that a hung connection fails the step rather than
#: sitting there until the job's own timeout, which reports nothing.
TIMEOUT_SECONDS = 300


def digest(archive: Path, *, block_size: int = BLOCK_SIZE) -> str:
    """Return the SHA-256 of a file, as lowercase hex.

    Args:
        archive: The file to hash.
        block_size: How many bytes to read at a time.

    Returns:
        The digest, in the spelling the recorded constants use.
    """
    hasher = hashlib.sha256()
    with archive.open("rb") as handle:
        while block := handle.read(block_size):
            hasher.update(block)
    return hasher.hexdigest()


def expected_problem(expected: str) -> str | None:
    """Report a recorded digest that is not a SHA-256 at all.

    Args:
        expected: The digest a driver records.

    Returns:
        A message, or ``None`` when it is well formed.
    """
    if not expected:
        return (
            "no SHA-256 was recorded for this archive. Every source archive "
            "this build extracts is pinned by content as well as by version "
            "(spec §10); a driver that names a URL has to name the digest of "
            "what that URL serves, next to the version it pins."
        )
    if len(expected) != DIGEST_LENGTH or not set(expected) <= DIGEST_ALPHABET:
        return (
            f"the recorded digest {expected!r} is not a SHA-256: it has to be "
            f"{DIGEST_LENGTH} lowercase hex characters, as `sha256sum` prints "
            "them. A truncated, uppercased or placeholder value would "
            "otherwise fail every run as though the archive had changed."
        )
    return None


def digest_problem(
    observed: str, expected: str, *, archive: Path, url: str = ""
) -> str | None:
    """Report an archive whose bytes are not the ones that were pinned.

    Args:
        observed: The digest of the file on disk.
        expected: The digest the driver records.
        archive: Where the file is, so the message can name it.
        url: Where it was fetched from, when the caller knows.

    Returns:
        A message naming both digests, or ``None`` when they agree.
    """
    malformed = expected_problem(expected)
    if malformed is not None:
        return malformed
    if observed == expected:
        return None
    source = f" fetched from {url}" if url else ""
    return (
        f"{archive}{source} is not the archive this build pinned:\n"
        f"  expected {expected}\n"
        f"  observed {observed}\n"
        "The version pin is satisfied and the bytes still changed, which is "
        "the case this check exists for: a re-rolled upstream tarball, a "
        "drifted mirror, or a digest that was not moved when the version was. "
        "The file has been left in place to be looked at rather than fetched "
        "again — a retry that happened to succeed would hide this."
    )


def verify(archive: Path, expected: str, *, url: str = "") -> str:
    """Prove an archive is the pinned one, before anything unpacks it.

    Args:
        archive: The downloaded file.
        expected: The digest the driver records.
        url: Where it was fetched from, when the caller knows.

    Returns:
        The archive's digest.

    Raises:
        ValueError: When the bytes are not the pinned ones, or when the
            recorded digest is not a SHA-256.
    """
    observed = digest(archive)
    problem = digest_problem(observed, expected, archive=archive, url=url)
    if problem is not None:
        raise ValueError(problem)
    return observed


class HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect that leaves ``https``.

    Checking the scheme of the URL that was asked for says nothing about the
    one that answers: a redirect to ``http`` would be followed silently. The
    digest check below would still catch altered bytes, but it would catch
    them after they had been read off a plaintext connection, and the point of
    refusing is that the sources are fetched over ``https`` at all.
    """

    def redirect_request(  # noqa: PLR0917 - the signature is urllib's
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Let urllib follow a redirect only while it stays on https."""
        if not newurl.startswith("https://"):
            raise ValueError(f"refusing to follow a redirect to {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


def download(url: str, archive: Path) -> Path:
    """Fetch an archive, leaving nothing behind if the fetch does not finish.

    A cache is reused across runs, so a half-written file is not one bad run —
    it is every later run reading the same truncated tarball out of the cache.
    With a digest check that failure at least says what it is, but it says it
    forever. So the download lands on a temporary name and is renamed only
    once it is complete, which on one filesystem is atomic.

    Args:
        url: Where to fetch from. Must be ``https``.
        archive: Where the finished file goes.

    Returns:
        The archive.

    Raises:
        ValueError: When the URL is not an HTTPS one, or redirects off it.
    """
    if not url.startswith("https://"):
        raise ValueError(f"{url} is not an https URL, and the sources are.")
    print(f"+ download {url}", flush=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(f"{archive.suffix}.part")
    request = urllib.request.Request(  # noqa: S310 - https, checked above
        url, headers={"User-Agent": USER_AGENT}
    )
    opener = urllib.request.build_opener(HttpsOnlyRedirects)
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            partial.write_bytes(response.read())
        partial.replace(archive)
    finally:
        partial.unlink(missing_ok=True)
    return archive


#: Written beside an archive whose *downloaded* bytes were refused, holding
#: the digest that was wanted and the digest that arrived. Its whole purpose
#: is to survive to the next run: without it, a refused download is
#: indistinguishable from a stale cache entry, and the retry it would get is
#: the one thing this module refuses to do.
REJECTION_SUFFIX = ".rejected"


def rejection(archive: Path) -> Path:
    """Return where an archive's rejection marker lives.

    Args:
        archive: The archive it would describe.

    Returns:
        The marker's path, whether or not it exists.
    """
    return archive.with_name(archive.name + REJECTION_SUFFIX)


def standing_refusal(archive: Path, expected: str, observed: str) -> str | None:
    """Report a refusal already recorded against exactly this situation.

    Args:
        archive: The file on disk.
        expected: The digest the driver records now.
        observed: What the file on disk hashes to now.

    Returns:
        A message, or ``None`` when there is no marker, when it describes
        some other digest, or when it cannot be read.
    """
    recorded = markers.text(rejection(archive))
    if recorded is None or recorded.split()[:2] != [expected, observed]:
        # Something it describes has moved: the driver's digest was bumped, or
        # the archive was replaced or deleted. Either way the refusal was
        # about a state that no longer exists.
        return None
    return (
        f"{archive} was downloaded and refused on an earlier run, and neither "
        f"it nor the recorded digest has changed since:\n"
        f"  expected {expected}\n"
        f"  observed {observed}\n"
        "It is not being fetched again. A retry that happened to succeed "
        "would turn the one signal this check exists to raise into a "
        "transient, so the same answer is given until something changes: "
        "correct the digest the driver records, or delete the archive to ask "
        f"upstream again. The marker is {rejection(archive)}."
    )


def fetch(url: str, archive: Path, expected: str) -> Path:
    """Return the pinned archive, downloading it unless the cache holds it.

    The verification is not conditional on the download: an archive already in
    a cache is hashed too, so a tampered one is caught on the run that unpacks
    it rather than on the run that fetched it. What it is not is *refused* for
    being cached — a cache entry that is merely not the pinned archive is
    replaced, once, and it is the downloaded bytes that are then judged
    (ticket 26). Bytes that this code downloaded and refused are the exception,
    and the marker beside them is what makes them one.

    Args:
        url: Where to fetch from.
        archive: Where the file is, or is to be put.
        expected: The digest the driver records.

    Returns:
        The archive, whose bytes are the pinned ones.

    Raises:
        ValueError: When the URL is not ``https``, the recorded digest is not
            a SHA-256, the downloaded bytes are not the pinned ones, or a
            refusal recorded on an earlier run still stands.
    """
    # Before anything is fetched: a digest that is a typo cannot be satisfied
    # by any bytes upstream serves, and saying so costs no download.
    malformed = expected_problem(expected)
    if malformed is not None:
        raise ValueError(malformed)
    marker = rejection(archive)
    if archive.exists():
        observed = digest(archive)
        if observed == expected:
            marker.unlink(missing_ok=True)
            return archive
        refusal = standing_refusal(archive, expected, observed)
        if refusal is not None:
            raise ValueError(refusal)
        # No standing refusal, so these are not bytes this code fetched and
        # judged — they are a cache entry from an earlier pin. The archive is
        # named for the release, so the previous release's bytes sit exactly
        # here after a digest bump; left refused they are refused forever,
        # because nothing else would ever overwrite them.
        print(
            f"+ replacing {archive}: it is not the pinned archive, and the "
            "cache is keyed by release rather than by content",
            flush=True,
        )
    download(url, archive)
    observed = digest(archive)
    problem = digest_problem(observed, expected, archive=archive, url=url)
    if problem is not None:
        # Recorded before raising, so the next run repeats this answer instead
        # of asking upstream again (ticket 26 restored ticket 10's rule here).
        markers.write(marker, f"{expected}\n{observed}")
        raise ValueError(problem)
    marker.unlink(missing_ok=True)
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: put the pinned archive where the shell asked.

    ``scripts/build-wheel.sh``'s ``fetch_source`` calls this instead of
    fetching for itself, so the rules above are written once rather than once
    per language (ticket 28). It used to ``curl`` unconditionally and call
    this as a front for :func:`verify`, which meant the shell path had the
    refusal rule nowhere: a refused archive was fetched again on every run,
    and a mismatch that cleared upstream between two runs passed on the
    second with nothing said about the first.

    Returning rather than raising is what the shell reads: a non-zero exit is
    the whole report, so the message goes to stderr and ``tar`` never runs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="where the driver names it")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--expected", required=True, help="the SHA-256 the driver records"
    )
    args = parser.parse_args(argv)

    try:
        archive = fetch(args.url, args.archive, args.expected)
    except ValueError as refused:
        print(f"ERROR: {refused}", file=sys.stderr)
        return 1
    except urllib.error.URLError as unreachable:
        # Before the OSError branch it would fall into: urllib's errors are
        # OSErrors, and reported as an unreadable archive they name a local
        # file that was never created. A 404 and a digest refusal are
        # different answers and the build log has to be able to tell them
        # apart -- which the `curl -fsSL` this replaced did for free.
        print(f"ERROR: cannot download {args.url}: {unreachable}", file=sys.stderr)
        return 1
    except OSError as unreadable:
        print(f"ERROR: cannot read {args.archive}: {unreadable}", file=sys.stderr)
        return 1

    print(f"{archive} is the pinned archive (sha256 {args.expected})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
