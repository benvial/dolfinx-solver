"""Read and write the small files this build leaves in its caches.

Three stages decide "already done" from a file rather than from the work: the
install prefix records the scalar variant that claimed it
(:mod:`wheelbuild.prefix`), a refused download leaves a rejection beside the
archive (:mod:`wheelbuild.sources`), and an extracted source tree records the
digest it came out of (:mod:`wheeltest.demos`). Each is written by a run that
can be killed at any point — a CI job hitting its timeout, an operator's
Ctrl-C — and read by a later run that has nothing else to go on.

That makes two rules, which :mod:`wheelbuild.prefix` arrived at first
(ticket 24) and which every marker written from Python now shares. The one
marker written from the shell — ``scripts/build-wheel.sh``'s own ``.extracted``
file, beside the source trees the container build unpacks — keeps its plain
``printf``: it is read with ``cat`` and compared, so bytes that do not decode
are a mismatch there rather than a failure, and a truncated one re-extracts.
The rules are:

* **A marker that cannot be read says nothing.** Absent, holding bytes that
  are not UTF-8, a directory where a file was expected: all the same answer,
  because the caller judges the *text* and a marker outside what it knows is
  ignored either way. Raising instead would put a ``UnicodeDecodeError``
  where the caller expects an answer, on a path whose whole design is that a
  later run repairs the earlier one — and since nothing rewrites a marker it
  refuses to read, the build would need a human to delete the file.
* **A marker is replaced, not truncated.** Writing in place empties the file
  first, so a run killed in the window between leaves a marker that says
  nothing where one said something. The text is written beside the marker and
  renamed over it instead, which within a single directory is atomic: an
  interrupted run leaves either the old marker or the new one, never half of
  one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: What the replacement is called while it is being written. It sits in the
#: marker's own directory, because a rename is atomic only within one
#: filesystem and that is the only directory a marker's writer is known to
#: have.
PENDING_SUFFIX = ".new"


def text(marker: Path) -> str | None:
    """Return what a marker says, or ``None`` when it says nothing.

    Args:
        marker: The file to read, which need not exist.

    Returns:
        The marker's text without surrounding whitespace, or ``None`` when it
        cannot be read at all.
    """
    try:
        return marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def write(marker: Path, content: str) -> None:
    """Record a marker in one step that either happens or does not.

    Args:
        marker: Where the marker goes. Its directory has to exist, which for
            every caller here is the cache or prefix the marker describes.
        content: What the marker is to say. The trailing newline is this
            function's, as the stripping in :func:`text` is — a caller that
            had to remember one and a reader that had to allow for one is two
            chances to disagree about the same file.

    Raises:
        OSError: When the marker cannot be written.
    """
    pending = marker.with_name(marker.name + PENDING_SUFFIX)
    try:
        pending.write_text(f"{content}\n", encoding="utf-8")
        pending.replace(marker)
    except BaseException:
        # The rename never happened, so the temporary is still there — and a
        # temporary left in a cache that is kept between runs is read by
        # nothing and cleaned up by nothing either.
        pending.unlink(missing_ok=True)
        raise
