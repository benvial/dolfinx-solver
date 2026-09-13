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

A mismatch is deliberately not repaired by re-downloading. A second fetch that
happened to succeed would turn the one signal this module exists to raise into
a transient, so the archive is left where it is and the message names it: the
question "are these the bytes we pinned?" is answered by looking at the file,
not by trying again.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

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


def fetch(url: str, archive: Path, expected: str) -> Path:
    """Return a verified archive, downloading it if it is not already there.

    The verification is not conditional on the download: an archive already in
    a cache is hashed too, so a tampered one is caught on the run that unpacks
    it rather than on the run that fetched it.

    Args:
        url: Where to fetch from.
        archive: Where the file is, or is to be put.
        expected: The digest the driver records.

    Returns:
        The archive.

    Raises:
        ValueError: When the URL is not ``https``, or the bytes are not the
            pinned ones.
    """
    if not archive.exists():
        download(url, archive)
    verify(archive, expected, url=url)
    return archive


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: verify an archive the shell downloaded.

    ``scripts/build-wheel.sh`` fetches with ``curl`` and calls this between
    the download and ``tar``, so nothing is extracted unverified.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--expected", required=True, help="the SHA-256 the driver records"
    )
    parser.add_argument("--url", default="", help="where it was fetched from")
    args = parser.parse_args(argv)

    try:
        observed = verify(args.archive, args.expected, url=args.url)
    except ValueError as mismatch:
        print(f"ERROR: {mismatch}", file=sys.stderr)
        return 1
    except OSError as unreadable:
        print(f"ERROR: cannot read {args.archive}: {unreadable}", file=sys.stderr)
        return 1

    print(f"{args.archive} is the pinned archive (sha256 {observed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
