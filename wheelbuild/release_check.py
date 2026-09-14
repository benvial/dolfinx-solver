"""Notice an upstream DOLFINx release and make the bump mechanically.

DOLFINx is not a package in any ecosystem Dependabot understands — this wheel
*is* the packaging of it — so nothing tells this repository that upstream has
tagged a release. The workflow that runs this module asks GitHub once a week,
and when upstream is ahead it opens the bump as a pull request.

A bump is more than a version string here. Four files carry the mirrored
release, and each is read by something that fails late when it is left behind:

* ``dolfinx_solver/_version.py`` — ``RELEASE``, ``DOLFINX_VERSION`` and
  ``__version__``, the single source the build drivers, ``pyproject.toml`` and
  the release tag all read (spec §3).
* ``wheelbuild/dolfinx.py`` — ``DOLFINX_SHA256``, the digest of the source
  tarball. New release, new bytes: the old digest stops the build at its first
  download, which is what it is for, and only a download can say what replaces
  it.
* ``pyproject.toml`` — the ``fenics-basix``, ``fenics-ffcx`` and
  ``fenics-ufl`` pins, which are upstream's own, copied verbatim out of the
  release's ``python/pyproject.toml`` (spec §10). Copying is mechanical in the
  strict sense — there is nothing to decide — and
  :mod:`wheelbuild.pin_check` fails CI when the copy is not exact.
* ``meta/pyproject.toml`` — the meta-package's version and the exact pin it
  puts on its variant. The three distributions release in lockstep, so a
  meta-package left at the old number is a release that cannot be completed.

What this module deliberately does not touch is
:data:`wheelbuild.dolfinx.NANOBIND_VERSION`, and neither does it touch
:data:`wheelbuild.mpich.MPICH_VERSION`. The first is a coupling to the
``fenics-basix`` wheel's nanobind ABI rather than to anything upstream
declares; the second has nothing to do with the DOLFINx release at all. Both
are decisions, and a weekly job does not get to make them: :func:`report`
writes down what the new release says about them and leaves them to a reader.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from wheelbuild import dolfinx, mpich, pin_check, publish, sources
from wheelbuild import version as version_module

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: The repository this wheel mirrors, as GitHub's API spells it.
UPSTREAM_REPOSITORY = "FEniCS/dolfinx"

#: Where the newest release is read from. ``/releases/latest`` skips drafts
#: and prereleases, so an upstream release candidate opens no pull request.
LATEST_RELEASE_URL = (
    f"https://api.github.com/repos/{UPSTREAM_REPOSITORY}/releases/latest"
)

#: Where one release's notes live, for the pull request to link to.
RELEASE_PAGE = f"https://github.com/{UPSTREAM_REPOSITORY}/releases/tag/v"

#: The version module inside the package.
VERSION_MODULE = version_module.VERSION_SOURCE

#: The driver that records the source digest.
DOLFINX_DRIVER = Path(dolfinx.__file__).resolve()

#: This project's metadata, which carries the trio pins.
PYPROJECT = pin_check.PYPROJECT_PATH

#: The meta-package's metadata, which releases in lockstep.
META_PYPROJECT = publish.META_PYPROJECT

#: What a mirrorable release looks like: upstream's number, with upstream's
#: own packaging segment when it has one. Anything else — an ``rc``, a
#: ``dev``, a local segment — is not something this packaging mirrors on its
#: own, because ``RELEASE`` is the version without a ``.postN`` and nothing
#: else may sit between the two (spec §3).
MIRRORABLE = re.compile(r"^\d+(?:\.\d+)*(?:\.post\d+)?$")


class Bump(NamedTuple):
    """One line a bump rewrote."""

    #: The file it is in, as the pull request names it.
    path: str

    #: The line before the bump, stripped.
    was: str

    #: The line after it, stripped.
    now: str


def release_of(dolfinx_version: str) -> str:
    """Return an upstream release with no packaging segment.

    Args:
        dolfinx_version: A release as upstream tags it, such as
            ``0.11.0.post0``.

    Returns:
        The release itself, such as ``0.11.0``: what DOLFINx's own
        ``project(DOLFINX VERSION ...)`` line carries, and therefore what the
        installed ``dolfinx.pc`` reports.
    """
    release, _, _ = dolfinx_version.partition(".post")
    return release


def mirrorable_problem(dolfinx_version: str) -> str | None:
    """Report an upstream release this packaging cannot mirror mechanically.

    Args:
        dolfinx_version: The release being bumped to, without its ``v``.

    Returns:
        A message, or ``None`` when the release is an ordinary one.
    """
    if MIRRORABLE.match(dolfinx_version) is None:
        return (
            f"{dolfinx_version!r} is not a release this packaging mirrors on "
            "its own: the version it publishes is upstream's number with a "
            "packaging .postN segment, and anything else between the two is a "
            "decision rather than a copy (spec §3). Bump it by hand."
        )
    return None


def is_behind(packaged: str, upstream: str) -> bool:
    """Report whether upstream has released something newer than we ship.

    Compared as versions rather than as strings, so a yanked release that
    takes upstream's ``latest`` backwards is not news, and ``.post10`` is
    correctly newer than ``.post9``.

    Args:
        packaged: The DOLFINx release this repository mirrors.
        upstream: The release upstream calls latest.

    Returns:
        ``True`` when the upstream release is the newer of the two.

    Raises:
        ValueError: When either side is not a version at all.
    """
    try:
        return Version(upstream) > Version(packaged)
    except InvalidVersion as unparsable:
        raise ValueError(f"not a version: {unparsable}") from unparsable


def latest_upstream_release(work_dir: Path | None = None) -> str:
    """Return the release upstream calls latest, without its ``v``.

    Args:
        work_dir: Directory to read the API response into. A temporary one by
            default.

    Returns:
        The tag name with its ``v`` stripped, such as ``0.11.0.post0``.

    Raises:
        OSError: When GitHub cannot be reached, or answers something that is
            not a release. Raised rather than swallowed: a job that decided
            "not behind" because it could not read the answer is a weekly
            check that has silently stopped working.
    """
    with TemporaryDirectory() as scratch:
        target = (work_dir or Path(scratch)) / "latest-release.json"
        sources.download(LATEST_RELEASE_URL, target)
        payload = json.loads(target.read_text(encoding="utf-8"))
    tag = payload.get("tag_name")
    if not isinstance(tag, str) or not tag:
        raise OSError(
            f"{LATEST_RELEASE_URL} answered no tag_name, so what upstream's "
            "latest release is could not be read."
        )
    return tag.removeprefix("v")


def fetch_release(dolfinx_version: str, work_dir: Path) -> tuple[Path, str]:
    """Download a release's source tarball and hash it.

    Downloaded rather than verified, because the digest is what is being
    learned: :data:`wheelbuild.dolfinx.DOLFINX_SHA256` pins bytes, and bytes
    nobody has fetched yet have no pin to check against. This is therefore the
    one fetch in the project that is trusted rather than checked, which is why
    it happens here — in a pull request a human reads — rather than in a
    build.

    Args:
        dolfinx_version: The release to fetch.
        work_dir: Directory to download into.

    Returns:
        The archive and its SHA-256.
    """
    archive = sources.download(
        dolfinx.source_url(dolfinx_version),
        work_dir / f"{dolfinx.source_dir_name(dolfinx_version)}.tar.gz",
    )
    return archive, sources.digest(archive)


def upstream_pyproject(archive: Path, dolfinx_version: str) -> str:
    """Read upstream's ``python/pyproject.toml`` out of a release tarball.

    Args:
        archive: The release's source tarball.
        dolfinx_version: The release it holds.

    Returns:
        The file's contents, which is where the trio pins are copied from.

    Raises:
        OSError: When the archive does not carry that file.
    """
    member = pin_check.upstream_pyproject_member(dolfinx_version)
    with tarfile.open(archive) as tar:
        try:
            extracted = tar.extractfile(member)
        except KeyError as missing:
            raise OSError(
                f"{archive} carries no {member}: the trio pins are copied "
                "from that file and from nothing else (spec §10)."
            ) from missing
        if extracted is None:
            raise OSError(f"{member} in {archive} is not a regular file.")
        return extracted.read().decode("utf-8")


def substitute(
    text: str,
    pattern: re.Pattern[str],
    replacement: str | Callable[[re.Match[str]], str],
    *,
    where: str,
) -> tuple[str, Bump | None]:
    """Rewrite the one line a pattern matches.

    Args:
        text: The file's contents.
        pattern: A pattern matching exactly the line to rewrite.
        replacement: The line it becomes, or a function of the match.
        where: The file, for the message and the reported bump.

    Returns:
        The rewritten text and what changed, or the text unchanged and
        ``None`` when it already said this.

    Raises:
        ValueError: When the pattern does not match exactly once. A
            substitution that quietly placed nothing is a pull request that
            looks like a bump and builds the old release.
    """
    found = list(pattern.finditer(text))
    if len(found) != 1:
        raise ValueError(
            f"{where} has {len(found)} lines matching {pattern.pattern!r} and "
            "the bump needs exactly one: the file and this driver have "
            "drifted apart."
        )
    match = found[0]
    line = replacement(match) if callable(replacement) else replacement
    if line == match.group(0):
        return text, None
    return (
        f"{text[: match.start()]}{line}{text[match.end() :]}",
        Bump(where, match.group(0).strip(), line.strip()),
    )


def bump_version_module(text: str, *, dolfinx_version: str) -> tuple[str, list[Bump]]:
    """Point the package's version module at a new upstream release.

    ``__version__`` becomes the upstream release exactly: the ``.postN``
    segment is upstream's own until a packaging-only fix moves it, and a fresh
    bump is not one.

    Args:
        text: Contents of ``dolfinx_solver/_version.py``.
        dolfinx_version: The release being mirrored.

    Returns:
        The rewritten module and the lines that changed.

    Raises:
        ValueError: When any of the three literals is not there exactly once.
    """
    changes = []
    for name, value in (
        ("RELEASE", release_of(dolfinx_version)),
        ("DOLFINX_VERSION", dolfinx_version),
        ("__version__", dolfinx_version),
    ):
        pattern = re.compile(rf'^{re.escape(name)} = "[^"]+"$', re.MULTILINE)
        text, change = substitute(
            text, pattern, f'{name} = "{value}"', where=VERSION_MODULE.name
        )
        if change is not None:
            changes.append(change)
    return text, changes


def bump_source_digest(text: str, *, digest: str) -> tuple[str, list[Bump]]:
    """Record the digest of the release's source tarball.

    Args:
        text: Contents of ``wheelbuild/dolfinx.py``.
        digest: SHA-256 of the archive :func:`wheelbuild.dolfinx.source_url`
            now serves.

    Returns:
        The rewritten driver and the line that changed.

    Raises:
        ValueError: When the digest is not a SHA-256, or the constant is not
            there exactly once.
    """
    malformed = sources.expected_problem(digest)
    if malformed is not None:
        raise ValueError(malformed)
    text, change = substitute(
        text,
        re.compile(r'^DOLFINX_SHA256 = "[^"]+"$', re.MULTILINE),
        f'DOLFINX_SHA256 = "{digest}"',
        where=DOLFINX_DRIVER.name,
    )
    return text, [change] if change is not None else []


def bump_trio_pins(text: str, upstream_pyproject_text: str) -> tuple[str, list[Bump]]:
    """Copy upstream's ``fenics-*`` pins into this project's metadata.

    Verbatim, because they answer a question only upstream can answer — which
    releases of Basix, FFCx and UFL this DOLFINx works with — and
    :func:`wheelbuild.pin_check.trio_problem` compares the copy back against
    the release's own file (spec §10).

    Args:
        text: Contents of this project's ``pyproject.toml``.
        upstream_pyproject_text: Contents of DOLFINx's
            ``python/pyproject.toml`` at the release being bumped to.

    Returns:
        The rewritten metadata and the pins that changed.

    Raises:
        ValueError: When upstream declares no pin for one of the three, or
            ours is not declared on exactly one line.
    """
    theirs = pin_check.declared_dependencies(upstream_pyproject_text)
    changes = []
    for name in pin_check.UPSTREAM_TRIO:
        upstream = theirs.get(canonicalize_name(name))
        if upstream is None:
            raise ValueError(
                f"the release being bumped to declares no {name} dependency, "
                "so there is no pin to copy: upstream restructured its "
                "metadata, and spec §10's coupling contract has to be re-read "
                "rather than re-derived."
            )

        def copied(match: re.Match[str], pin: str = upstream) -> str:
            """Return the pin, indented as the line it replaces was."""
            return f'{match.group("indent")}"{pin}",'

        text, change = substitute(
            text,
            re.compile(rf'^(?P<indent>\s*)"{re.escape(name)}[^"]*",$', re.MULTILINE),
            copied,
            where=PYPROJECT.name,
        )
        if change is not None:
            changes.append(change)
    return text, changes


def bump_mirrored_release(text: str, *, dolfinx_version: str) -> tuple[str, list[Bump]]:
    """Move the comment naming the release the trio pins were copied from.

    A comment rather than a value, and still worth moving: it is what tells
    the next reader which release to re-read the pins against.

    Args:
        text: Contents of this project's ``pyproject.toml``.
        dolfinx_version: The release being mirrored.

    Returns:
        The rewritten metadata and the line that changed.

    Raises:
        ValueError: When the comment is not there exactly once.
    """
    text, change = substitute(
        text,
        re.compile(
            r"^# python/pyproject\.toml at the mirrored release "
            r"\(v(?P<release>[^)]+)\)",
            re.MULTILINE,
        ),
        f"# python/pyproject.toml at the mirrored release (v{dolfinx_version})",
        where=PYPROJECT.name,
    )
    return text, [change] if change is not None else []


def bump_meta_project(text: str, *, wheel_version: str) -> tuple[str, list[Bump]]:
    """Keep the meta-package releasing in lockstep with the wheels.

    Its version and the exact pin it puts on its variant are the same number,
    and :mod:`wheelbuild.publish` refuses a release whose three distributions
    do not all carry it.

    Args:
        text: Contents of ``meta/pyproject.toml``.
        wheel_version: The version the wheels will carry.

    Returns:
        The rewritten metadata and the lines that changed.

    Raises:
        ValueError: When either literal is not there exactly once.
    """
    where = f"{META_PYPROJECT.parent.name}/{META_PYPROJECT.name}"
    changes = []
    for pattern, replacement in (
        (
            re.compile(r'^version = "[^"]+"$', re.MULTILINE),
            f'version = "{wheel_version}"',
        ),
        (
            re.compile(
                r'^dependencies = \["(?P<variant>dolfinx-solver-\w+)==[^"]+"\]$',
                re.MULTILINE,
            ),
            lambda match: (
                f'dependencies = ["{match.group("variant")}=={wheel_version}"]'
            ),
        ),
    ):
        text, change = substitute(text, pattern, replacement, where=where)
        if change is not None:
            changes.append(change)
    return text, changes


def upstream_nanobind_requirement(upstream_pyproject_text: str) -> str | None:
    """Return what a release's bindings build asks of nanobind.

    Args:
        upstream_pyproject_text: Contents of DOLFINx's
            ``python/pyproject.toml``.

    Returns:
        The requirement upstream declares, or ``None`` when it declares none.
    """
    requires = (
        tomllib.loads(upstream_pyproject_text)
        .get("build-system", {})
        .get("requires", [])
    )
    return next(
        (
            requirement
            for requirement in requires
            if canonicalize_name(Requirement(requirement).name) == "nanobind"
        ),
        None,
    )


def bump(
    dolfinx_version: str, *, work_dir: Path, root: Path | None = None
) -> tuple[list[Bump], str]:
    """Write the release into every file that carries it.

    Args:
        dolfinx_version: The release to mirror.
        work_dir: Directory to download the source tarball into.
        root: Repository root, for tests that bump a copy.

    Returns:
        The lines that changed and the contents of upstream's
        ``python/pyproject.toml`` at that release, which the report reads.

    Raises:
        ValueError: When the release cannot be mirrored mechanically, or a
            file no longer says what the bump rewrites.
    """
    unmirrorable = mirrorable_problem(dolfinx_version)
    if unmirrorable is not None:
        raise ValueError(unmirrorable)

    base = root or PYPROJECT.parent
    archive, digest = fetch_release(dolfinx_version, work_dir)
    upstream_text = upstream_pyproject(archive, dolfinx_version)

    changes: list[Bump] = []
    for path, edits in (
        (
            base / VERSION_MODULE.relative_to(PYPROJECT.parent),
            [lambda text: bump_version_module(text, dolfinx_version=dolfinx_version)],
        ),
        (
            base / DOLFINX_DRIVER.relative_to(PYPROJECT.parent),
            [lambda text: bump_source_digest(text, digest=digest)],
        ),
        (
            base / PYPROJECT.name,
            [
                lambda text: bump_trio_pins(text, upstream_text),
                lambda text: bump_mirrored_release(
                    text, dolfinx_version=dolfinx_version
                ),
            ],
        ),
        (
            base / META_PYPROJECT.relative_to(PYPROJECT.parent),
            [lambda text: bump_meta_project(text, wheel_version=dolfinx_version)],
        ),
    ):
        text = path.read_text(encoding="utf-8")
        for edit in edits:
            text, placed = edit(text)
            changes.extend(placed)
        path.write_text(text, encoding="utf-8")
    return changes, upstream_text


def report(
    *,
    packaged: str,
    upstream: str,
    changes: Sequence[Bump],
    upstream_pyproject_text: str,
) -> str:
    """Write the pull request body.

    Args:
        packaged: The release the repository mirrored before the bump.
        upstream: The release it mirrors after it.
        changes: The lines the bump rewrote.
        upstream_pyproject_text: DOLFINx's own metadata at the new release.

    Returns:
        Markdown: what moved, what CI will do with it, and what is left for a
        human to decide.
    """
    rewritten = "\n".join(
        f"- `{change.path}`: `{change.was}` → `{change.now}`" for change in changes
    )
    nanobind = upstream_nanobind_requirement(upstream_pyproject_text)
    asked = f"`{nanobind}`" if nanobind else "nothing"
    return f"""\
Upstream released [DOLFINx {upstream}]({RELEASE_PAGE}{upstream}). This mirrors
it: {packaged} is what the repository shipped before.

## What moved

{rewritten}

The trio pins are copied verbatim out of the release's own
`python/pyproject.toml` rather than chosen (spec §10), and the source digest is
this job's SHA-256 of the tarball GitHub served for the tag — the one fetch in
the project that is trusted rather than checked, which is why it is in a diff
you can read.

## What CI proves here

Everything a tag runs except the upload: lint, mypy, the unit tests,
`wheelbuild.pin_check` against the new release's own file, then both variants
built in the manylinux image and put through the whole `wheeltest` suite on
3.12, 3.13 and 3.14. The superbuild cache key carries `_version.py` and
`pyproject.toml`, so this run misses it and falls back to the previous prefix
through the restore key: the vendored stack is reused and the DOLFINx stages
rebuild against the new sources. A green run means the wheel builds, imports
and solves.

## What a human still decides

- **`NANOBIND_VERSION`.** The release's bindings build requires {asked}; this
  repository pins `{dolfinx.NANOBIND_VERSION}` instead, because what the
  extension has to agree with is the nanobind ABI tag inside the
  `fenics-basix` wheel the new trio pins resolve to, not the floor upstream
  declares. `wheelbuild.dolfinx.nanobind_problem` fails the build if they
  disagree; moving the pin is a decision, so this job leaves it alone.
- **`MPICH_VERSION`.** Independent of the DOLFINx release and never bumped
  here. Moving it means re-running the interop stage, which
  `wheelbuild.pin_check` guards — MPICH {mpich.MPICH_VERSION} is what this
  wheel vendors the Fortran half of.
- **Upstream's API changes.** The smoke problem and the demos are upstream's
  own code paths; a green suite says they still run, not that the release
  notes hold nothing worth knowing. Read them before merging.
- **The tag.** Deliberately not automated: pushing `v{upstream}` publishes
  three distributions to PyPI through trusted publishing, and PyPI never lets
  a filename be reused.
"""


def emit(name: str, value: str) -> None:
    """Report one fact to the step that runs this, and to the log.

    Args:
        name: The output's name.
        value: Its value.
    """
    print(f"{name}={value}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the release check.

    With no arguments it reads upstream's latest release and says whether this
    repository is behind it, which is all the scheduled job needs to decide
    whether to go on. ``--apply`` then writes the release into the four files
    that carry it and, with ``--report``, the pull request body.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        help="the release to compare against (default: upstream's latest)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the release into the files that carry it",
    )
    parser.add_argument("--report", type=Path, help="where to write the bump's body")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="directory to download the source tarball into",
    )
    args = parser.parse_args(argv)

    packaged = version_module.DOLFINX_VERSION
    try:
        upstream = (
            args.upstream.removeprefix("v")
            if args.upstream
            else (latest_upstream_release())
        )
    except OSError as unreachable:
        print(f"ERROR: {unreachable}", file=sys.stderr)
        return 1

    emit("packaged", packaged)
    emit("upstream", upstream)
    behind = is_behind(packaged, upstream)
    emit("behind", "true" if behind else "false")
    if not behind:
        print(f"DOLFINx {packaged} is the release upstream calls latest")
        return 0
    if not args.apply:
        return 0

    with TemporaryDirectory() as scratch:
        work_dir = args.work_dir or Path(scratch)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            changes, upstream_text = bump(upstream, work_dir=work_dir)
        except (OSError, ValueError) as refused:
            print(f"ERROR: {refused}", file=sys.stderr)
            return 1

    for change in changes:
        print(f"+ {change.path}: {change.now}")
    if args.report is not None:
        args.report.write_text(
            report(
                packaged=packaged,
                upstream=upstream,
                changes=changes,
                upstream_pyproject_text=upstream_text,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
