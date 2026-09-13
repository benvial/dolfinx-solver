"""Keep this wheel's declared pins and the things they stand for in step.

Two pins in ``pyproject.toml`` are not free choices, and neither of them has
anything in the packaging machinery that keeps it honest. This module is that
tie, and it needs seconds rather than the hours the container build takes, so
CI runs it in the fast-fail checks job.

**The ``mpich`` bound** names the MPICH series the vendored ``libmpifort`` was
built against.

The wheel vendors ``libmpifort`` and depends on the PyPI ``mpich`` wheel for
``libmpi.so.12``. Those two halves are linked at the user's machine by a
soname that MPICH freezes across major series, so the resolver has no way to
tell a compatible pairing from an incompatible one — that job belongs to the
``mpich`` requirement in ``pyproject.toml``, which has to name exactly the
series ``wheelbuild.mpich`` builds (ADR-0001).

Nothing keeps the two numbers together on its own. Bumping
``MPICH_VERSION`` without the pin publishes a wheel whose Fortran half is
newer than the ``libmpi`` a fresh install resolves; loosening the pin to a
bare floor admits the same failure from the other direction. This check is
that tie, and it needs nothing but this repository, so CI runs it on every
push — the symbol-level proof of the same pairing runs in the container build
(``wheelbuild.mpich.interop_problem``), hours later.

The bound is derived rather than configured: one series, spelled
``>=<major>.0,<<major+1>``, so there is no second place to keep correct.

**The upstream trio pins** — ``fenics-basix``, ``fenics-ffcx``, ``fenics-ufl``
— are upstream DOLFINx's own, copied verbatim from its ``python/pyproject.toml``
at the release this wheel mirrors (spec §10). The trio is never re-vendored: the
C++ core in this wheel compiles against Basix's headers and FFCx's ``ufcx.h``,
and which releases of those a given DOLFINx works with is a question only
upstream can answer. Copying a pin is how the answer gets here, and a copy has
no way of noticing that the original changed — an upstream ``.post`` release
that widens or moves a bound leaves our metadata admitting a trio DOLFINx does
not support, and the failure lands in a user's resolver rather than here.
:func:`trio_problem` is the comparison; it reads upstream's file out of the
source tarball of the mirrored release, verified against the digest
:mod:`wheelbuild.dolfinx` records for it (spec §10), which is the same text
the container build compiles DOLFINx from.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tomllib
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from wheelbuild import sources
from wheelbuild.mpich import MPICH_VERSION, series
from wheelbuild.version import DOLFINX_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The distribution the runtime ``libmpi`` comes from.
MPICH_DISTRIBUTION = "mpich"

#: This project's metadata, where the pins are declared.
PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"

#: The pure-Python FEniCS packages DOLFINx is built and run against, whose
#: pins are upstream's rather than ours (spec §10). Named here rather than in
#: :mod:`wheelbuild.dolfinx` because this is where they are checked, and that
#: module already imports this one.
UPSTREAM_TRIO = ("fenics-basix", "fenics-ffcx", "fenics-ufl")

#: Where upstream declares them, inside the DOLFINx source tree.
UPSTREAM_PYPROJECT = "python/pyproject.toml"


def upstream_pyproject_member(dolfinx_version: str = DOLFINX_VERSION) -> str:
    """Return the path of upstream's ``pyproject.toml`` inside its tarball.

    Args:
        dolfinx_version: The DOLFINx release this wheel mirrors.

    Returns:
        The archive member's name, as upstream's GitHub tarball spells it.
    """
    return f"dolfinx-{dolfinx_version}/{UPSTREAM_PYPROJECT}"


def fetch_upstream_pyproject(cache: Path | None = None) -> tuple[str, str]:
    """Read upstream's ``pyproject.toml`` out of the release it ships in.

    Not off ``raw.githubusercontent.com``, which is where this used to read it
    from. That file decides whether a check passes, and nothing verified it:
    the tag made the content immutable in practice and that was the whole of
    the argument. The source tarball of the same release is pinned by content
    (:data:`wheelbuild.dolfinx.DOLFINX_SHA256`, ticket 10), it is under a
    megabyte, and it is the same archive the container build compiles DOLFINx
    from — so the trio pins are now compared against bytes this repository has
    checked, and the comparison still costs seconds rather than the hours the
    build takes.

    Args:
        cache: Directory to download into. A temporary one by default, since
            CI has no reason to keep it between runs.

    Returns:
        The file's contents and a description of where they were read from.

    Raises:
        ValueError: When the archive is not the pinned one.
        OSError: When it cannot be fetched or does not carry the file.
            Raised rather than swallowed: a pin check that passes because it
            could not reach the thing it compares against is worse than one
            that was never written.
    """
    # Imported here rather than at module scope: wheelbuild.dolfinx reads
    # UPSTREAM_TRIO from this module, so the two cannot import each other at
    # import time.
    from wheelbuild import dolfinx  # noqa: PLC0415

    with ExitStack() as scope:
        if cache is None:
            cache = Path(scope.enter_context(TemporaryDirectory()))
        url = dolfinx.source_url()
        try:
            archive = sources.fetch(
                url,
                cache / f"{dolfinx.source_dir_name()}.tar.gz",
                dolfinx.DOLFINX_SHA256,
            )
        except OSError as unreachable:
            # urllib's own errors name the host at best, and a check that
            # could not run has to say what it could not reach.
            raise OSError(f"{url} could not be fetched: {unreachable}") from unreachable
        member = upstream_pyproject_member()
        with tarfile.open(archive) as tar:
            try:
                extracted = tar.extractfile(member)
            except KeyError as missing:
                raise OSError(
                    f"{url} carries no {member}, so upstream's own trio pins "
                    "could not be read out of the release this wheel mirrors."
                ) from missing
            if extracted is None:
                raise OSError(f"{member} in {url} is not a regular file.")
            return extracted.read().decode("utf-8"), f"{url}::{member}"
    raise AssertionError  # pragma: no cover - ExitStack always yields


def declared_dependencies(pyproject_text: str) -> dict[str, str]:
    """Return a project's runtime dependencies, keyed by canonical name.

    Args:
        pyproject_text: Contents of a ``pyproject.toml``.

    Returns:
        A mapping from the PEP 503 normalised project name to the requirement
        string exactly as it was written.
    """
    dependencies = (
        tomllib.loads(pyproject_text).get("project", {}).get("dependencies", [])
    )
    return {
        canonicalize_name(Requirement(dependency).name): dependency
        for dependency in dependencies
    }


def _comparable(requirement: str) -> tuple[SpecifierSet, frozenset[str], str | None]:
    """Return what has to match for two requirements to install the same thing.

    The project name is deliberately left out: the caller has already matched
    the two on their canonical names, and PEP 503 lets each side spell it
    differently without meaning anything different.

    Args:
        requirement: A requirement string.

    Returns:
        Its version specifier, its extras and its environment marker.
    """
    parsed = Requirement(requirement)
    marker = None if parsed.marker is None else str(parsed.marker)
    return parsed.specifier, frozenset(parsed.extras), marker


#: The distribution upstream's ``python/pyproject.toml`` builds. Read back
#: from the file being compared against, so that a path pointing at the wrong
#: project's metadata is a failure rather than three missing dependencies.
UPSTREAM_DISTRIBUTION = "fenics-dolfinx"


def upstream_release_problem(upstream_pyproject_text: str) -> str | None:
    """Report an upstream ``pyproject.toml`` that is not the one to copy from.

    ``--upstream-pyproject`` points at a file on disk, and a source tree left
    over from a previous release is exactly the kind of thing that is still
    lying around. Comparing against it would prove our pins match a DOLFINx
    this wheel does not ship, and pass.

    Args:
        upstream_pyproject_text: Contents of the file being compared against.

    Returns:
        A message, or ``None`` when the file is DOLFINx's at the mirrored
        release.
    """
    project = tomllib.loads(upstream_pyproject_text).get("project", {})
    name = project.get("name")
    if name is None or canonicalize_name(name) != canonicalize_name(
        UPSTREAM_DISTRIBUTION
    ):
        return (
            f"the upstream metadata names {name!r}, not "
            f"{UPSTREAM_DISTRIBUTION!r}. The trio pins are copied from "
            "DOLFINx's own python/pyproject.toml and from nothing else "
            "(spec §10)."
        )
    version = project.get("version")
    if version != DOLFINX_VERSION:
        return (
            f"the upstream metadata is DOLFINx {version!r}, but this wheel "
            f"ships {DOLFINX_VERSION!r}. Pins copied from another release "
            "say nothing about the one being built; point the check at the "
            "mirrored tag."
        )
    return None


def trio_problem(pyproject_text: str, upstream_pyproject_text: str) -> str | None:
    """Report a trio pin of ours that is not the one upstream declares.

    Args:
        pyproject_text: Contents of this project's ``pyproject.toml``.
        upstream_pyproject_text: Contents of upstream DOLFINx's
            ``python/pyproject.toml`` at the mirrored release.

    Returns:
        A message naming the first project whose pins disagree, or ``None``
        when all three are copied as upstream wrote them.
    """
    wrong_release = upstream_release_problem(upstream_pyproject_text)
    if wrong_release is not None:
        return wrong_release

    ours = declared_dependencies(pyproject_text)
    theirs = declared_dependencies(upstream_pyproject_text)

    for name in UPSTREAM_TRIO:
        key = canonicalize_name(name)
        upstream = theirs.get(key)
        if upstream is None:
            return (
                f"upstream DOLFINx {DOLFINX_VERSION} declares no {name} "
                "dependency, so there is no pin here to copy. Either the "
                "release being mirrored is not the one this check is reading, "
                "or upstream restructured its metadata and spec §10's "
                "coupling contract has to be re-read rather than re-derived."
            )
        declared = ours.get(key)
        if declared is None:
            return (
                f"no {name} dependency is declared in pyproject.toml, but "
                f"upstream DOLFINx {DOLFINX_VERSION} declares "
                f"{upstream!r}. The C++ core in this wheel is compiled "
                "against that release; without the pin, pip is free to "
                "install another one beside it (spec §10)."
            )
        # The whole requirement bar its spelling of the name: a marker or an
        # extra changes what pip installs just as a specifier does, and
        # `fenics-ffcx>=0.11.0,<0.12.0 ; python_version < "3.13"` would
        # otherwise compare equal to the unconditional pin upstream declares.
        if _comparable(declared) != _comparable(upstream):
            return (
                f"the declared {declared!r} is not upstream's pin: DOLFINx "
                f"{DOLFINX_VERSION} declares {upstream!r}. Upstream's pins "
                "are the coupling contract between the C++ core this wheel "
                "vendors and the pure-Python trio it is compiled against, and "
                "they are copied verbatim rather than chosen (spec §10). Copy "
                "the pin across, or mirror a different release."
            )
    return None


def declared_mpich_requirement(pyproject_text: str) -> str:
    """Return the ``mpich`` requirement a ``pyproject.toml`` declares.

    Args:
        pyproject_text: Contents of the project's ``pyproject.toml``.

    Returns:
        The requirement string, such as ``mpich>=5.0,<6``.

    Raises:
        LookupError: When the project declares no dependency on mpich, which
            would leave the runtime libmpi to chance.
    """
    # A requirement may spell the project any way PEP 503 normalises to the
    # same name, so the lookup is on the canonical form.
    declared = declared_dependencies(pyproject_text).get(MPICH_DISTRIBUTION)
    if declared is not None:
        return declared
    raise LookupError(
        f"no {MPICH_DISTRIBUTION} dependency is declared: this wheel vendors "
        "only the Fortran half of MPI and needs the PyPI mpich wheel to "
        "supply libmpi at run time (spec §5)."
    )


def expected_specifier(version: str) -> str:
    """Return the version range that names one MPICH series.

    Args:
        version: The MPICH release the wheel is built against.

    Returns:
        A specifier such as ``>=5.0,<6``: a floor at the start of the series
        and a ceiling at the next one.
    """
    major = int(series(version))
    return f">={major}.0,<{major + 1}"


def pin_problem(declared_requirement: str, vendored_version: str) -> str | None:
    """Report a declared ``mpich`` pin that is not the series we build against.

    Args:
        declared_requirement: The requirement from ``pyproject.toml``.
        vendored_version: The MPICH release ``wheelbuild.mpich`` builds.

    Returns:
        A message, or ``None`` when the pin names exactly that series.
    """
    requirement = Requirement(declared_requirement)
    expected = expected_specifier(vendored_version)

    if canonicalize_name(requirement.name) != MPICH_DISTRIBUTION:
        return (
            f"the declared requirement is on {requirement.name!r}, not "
            f"{MPICH_DISTRIBUTION!r}. The vendored libmpifort is MPICH's, and "
            "its symbols come from an MPICH libmpi; no other MPI family can "
            "supply them (spec §5)."
        )

    if requirement.specifier == SpecifierSet(expected):
        return None
    return (
        f"the declared pin {declared_requirement!r} is not the series this "
        f"build vendors: wheelbuild.mpich builds MPICH {vendored_version}, so "
        f"the pin has to be {MPICH_DISTRIBUTION}{expected}. A looser bound "
        "admits a libmpi older than the libmpifort in the wheel, which fails "
        "at the user's first import rather than at install; a tighter one "
        "fights the resolver over the wheel's routine .post releases "
        "(ADR-0001)."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the pin check.

    The mpich half needs nothing but this repository and so always runs. The
    trio half needs upstream's file, which is either read out of the pinned
    source tarball, downloaded and verified against its recorded digest
    (``--fetch-upstream``, what CI's checks job does), or read from a source
    tree already on disk (``--upstream-pyproject``, for the build container,
    which has that same verified release unpacked). With neither, it is
    skipped and says so: a check that quietly passed when it could not run
    would be worse than one that was never written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT_PATH)
    parser.add_argument(
        "--upstream-pyproject",
        type=Path,
        help="upstream DOLFINx's python/pyproject.toml, already on disk",
    )
    parser.add_argument(
        "--fetch-upstream",
        action="store_true",
        help="read upstream's python/pyproject.toml out of the pinned tarball",
    )
    args = parser.parse_args(argv)

    pyproject_text = args.pyproject.read_text(encoding="utf-8")

    try:
        declared = declared_mpich_requirement(pyproject_text)
    except LookupError as undeclared:
        print(f"ERROR: {undeclared}", file=sys.stderr)
        return 1

    problem = pin_problem(declared, MPICH_VERSION)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print(f"the declared {declared!r} names the MPICH {MPICH_VERSION} series")

    if args.upstream_pyproject is not None:
        upstream_text = args.upstream_pyproject.read_text(encoding="utf-8")
        source = str(args.upstream_pyproject)
    elif args.fetch_upstream:
        try:
            upstream_text, source = fetch_upstream_pyproject()
        except (OSError, ValueError) as unreachable:
            # Not a pass: the whole point of this half is that our pins are
            # only correct relative to a file we do not control, and not
            # having read it is not evidence about them. A digest mismatch
            # lands here too, and is the louder of the two.
            print(
                f"ERROR: could not read the upstream pins: {unreachable}",
                file=sys.stderr,
            )
            return 1
    else:
        print(
            "the upstream trio pins were not checked: pass --fetch-upstream "
            "or --upstream-pyproject to compare them against DOLFINx "
            f"{DOLFINX_VERSION}'s own"
        )
        return 0

    problem = trio_problem(pyproject_text, upstream_text)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print(f"the declared trio pins are the ones {source} declares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
