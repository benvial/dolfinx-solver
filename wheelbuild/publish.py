"""Gather what a release publishes, and refuse a set that is not it.

A tag produces three distributions, and they come from three different
places: the two variant wheels are artefacts of the wheel jobs, compiled
hours earlier in a container, and the bare ``dolfinx-solver`` meta-package is
built here from ``meta/`` because it has no binaries and nothing else in the
workflow builds it. This module is what puts them in one directory and then
reads that directory back before anything is uploaded.

Reading it back is the point. PyPI never reuses a filename, so every mistake
this could make is permanent:

* **A wheel from the wrong wheelhouse.** A finished build has two. The one
  under the install prefix holds the petsc4py, slepc4py and fenics_dolfinx
  wheels the earlier stages built for themselves, and publishing any of them
  would put a PETSc on PyPI under a name upstream owns (spec §6, §10). So the
  check is not "are ours here" but "is anything else here".
* **A release missing a variant.** The workflow builds and tests both
  (ticket 21), and a tag that uploads one of them publishes a version of
  ``dolfinx-solver-real`` that no ``dolfinx-solver-complex`` of the same
  number exists beside — a half release that cannot be completed afterwards
  under the same version.
* **A version that is not this one.** The artefacts are downloaded from a
  workflow run, and a directory is a thing that can hold what a previous
  step left in it. The tag check already tied the tag to
  ``dolfinx_solver.__version__`` before the build started; this ties the
  files.

The variants are read from :data:`wheelbuild.petsc.SCALAR_TYPES` rather than
from ``SCALAR_TYPE``: every other driver in this package is running for one
declared variant, and this one is the single place that is running for all of
them at once.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from packaging.requirements import Requirement
from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)
from packaging.version import Version

from wheelbuild import petsc
from wheelbuild._process import check_call
from wheelbuild.version import __version__

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Repository root, which holds the meta-package's project directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The meta-package's project directory. It is built here rather than in the
#: wheel job because it has no binaries: a pure-python wheel and an sdist out
#: of a ``pyproject.toml`` that names one dependency (spec §1).
META_PROJECT = REPO_ROOT / "meta"

#: The meta-package's own metadata, which is where the pin that decides what
#: ``pip install dolfinx-solver`` means is actually written.
META_PYPROJECT = META_PROJECT / "pyproject.toml"

#: The bare name, as PyPI spells it. Not a variant: it depends on one.
META_DISTRIBUTION = "dolfinx-solver"

#: The two binary distributions, one per scalar variant (spec §6). Both, not
#: the declared one — a release is the pair.
VARIANT_DISTRIBUTIONS = tuple(
    f"{META_DISTRIBUTION}-{variant}" for variant in petsc.SCALAR_TYPES
)

#: Everything a ``v*`` tag uploads, in the order the report lists it.
DISTRIBUTIONS = (*VARIANT_DISTRIBUTIONS, META_DISTRIBUTION)

#: What a built distribution's file name ends in, as against an sdist's.
WHEEL_SUFFIX = ".whl"


class Index(NamedTuple):
    """One index this project publishes to, from both ends.

    The two URLs are different services and neither can be derived from the
    other, so an index is the pair rather than a name plus a convention.

    Attributes:
        upload: Where a distribution is uploaded to.
        simple: Where it is installed back from, or ``None`` for pip's own
            default.
    """

    upload: str
    simple: str | None


#: Where a release can go. The real one is reached by a tag; the other is the
#: one-time dry run spec §9 asks for before the first release, and is why
#: both ends of this project's release path say which index they are talking
#: about rather than assuming.
INDEXES = {
    "pypi": Index(upload="https://upload.pypi.org/legacy/", simple=None),
    "testpypi": Index(
        upload="https://test.pypi.org/legacy/",
        simple="https://test.pypi.org/simple/",
    ),
}

#: Where a dry run's *dependencies* come from. TestPyPI holds this project's
#: uploads and nothing else: mpich, mpi4py, numpy, cffi and the upstream trio
#: are still on PyPI, so an install from the dry-run index needs this beside
#: it.
FALLBACK_SIMPLE_URL = "https://pypi.org/simple/"


def pinned_distribution(pyproject: Path = META_PYPROJECT) -> str:
    """Return the variant the meta-package resolves to.

    Read out of ``meta/pyproject.toml`` rather than derived from the default
    scalar type: what ``pip install dolfinx-solver`` means is that one
    dependency line, and a release path that asserted the derived answer
    would agree with itself while disagreeing with what it published.

    Args:
        pyproject: The meta-package's metadata.

    Returns:
        The distribution it depends on, normalised.

    Raises:
        ValueError: When it does not depend on exactly one thing. The
            meta-package is a name and a pin; anything else is a decision
            nobody recorded.
    """
    dependencies = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    if len(dependencies) != 1:
        raise ValueError(
            f"{pyproject} declares {len(dependencies)} dependencies, and the "
            "meta-package is one pin: which variant `pip install "
            f"{META_DISTRIBUTION}` resolves to has to be readable from it "
            "(spec §1)."
        )
    return str(canonicalize_name(Requirement(dependencies[0]).name))


class Gathered(NamedTuple):
    """One file gathered for the upload, as its own name describes it.

    Attributes:
        path: The file.
        distribution: The normalised name it declares.
        version: The version it declares.
    """

    path: Path
    distribution: str
    version: str


def read_file_name(path: Path) -> Gathered | None:
    """Return what a file name declares about itself.

    Args:
        path: A wheel or an sdist, named as the tool that built it names
            them.

    Returns:
        The distribution and version, or ``None`` when the name is neither a
        wheel's nor an sdist's.
    """
    try:
        name, version, _, _ = parse_wheel_filename(path.name)
    except InvalidWheelFilename:
        try:
            name, version = parse_sdist_filename(path.name)
        except InvalidSdistFilename:
            return None
    return Gathered(path, str(name), str(version))


def read_gathered(
    paths: Iterable[Path], distributions: Sequence[str]
) -> tuple[list[Gathered], str | None]:
    """Read every gathered file's name, refusing one that is not ours.

    The check is not "are ours here" but "is anything else here": the upload
    sends the whole directory, so a file nobody meant to publish is published.

    Args:
        paths: Everything gathered for the upload.
        distributions: What a release publishes.

    Returns:
        What was read, and a message about the first file that is not one of
        this project's — ``None`` when they all are.
    """
    gathered: list[Gathered] = []
    for path in sorted(paths):
        read = read_file_name(path)
        if read is None:
            return gathered, (
                f"{path.name} is neither a wheel nor an sdist. Only the "
                "files this release publishes may be in the directory the "
                "upload reads, because the upload reads all of it."
            )
        if read.distribution not in distributions:
            return gathered, (
                f"{path.name} is a {read.distribution} distribution, which "
                "this project does not publish. A finished build holds a "
                "second wheelhouse of petsc4py, slepc4py and fenics_dolfinx "
                "wheels built for the build's own use, and those carry names "
                "upstream owns (spec §6, §10)."
            )
        gathered.append(read)
    return gathered, None


def version_problem(gathered: Iterable[Gathered], version: str) -> str | None:
    """Report a gathered file that belongs to another release.

    Args:
        gathered: What was read from the directory the upload reads.
        version: The version this release is, from the package.

    Returns:
        A message, or ``None`` when every file carries this version. The
        comparison is between parsed versions rather than strings: a file
        name is written by the tool that built it, always in the normalised
        spelling, while ``__version__`` is written by hand. The tag check is
        where the literal spelling is the subject
        (:mod:`wheelbuild.tag_check`).
    """
    released = Version(version)
    for read in gathered:
        if Version(read.version) != released:
            return (
                f"{read.path.name} carries version {read.version}, and this "
                f"release is {version}. The tag check tied the tag to that "
                "number before the build started; a file from another run "
                "would publish it under this one."
            )
    return None


def completeness_problem(
    gathered: Iterable[Gathered], distributions: Sequence[str]
) -> str | None:
    """Report a release that is not one wheel from each distribution.

    Args:
        gathered: What was read from the directory the upload reads.
        distributions: What a release publishes.

    Returns:
        A message, or ``None`` when each publishes exactly one wheel, with an
        sdist only where one belongs.
    """
    gathered = list(gathered)
    for name in distributions:
        files = [read.path for read in gathered if read.distribution == name]
        wheels = [path for path in files if path.suffix == WHEEL_SUFFIX]
        if not wheels:
            return (
                f"no {name} wheel was gathered, and a release is one wheel "
                f"from each of {', '.join(distributions)}. The two variants "
                "are one release at one version, and a tag that publishes "
                "one of them cannot publish the other afterwards under the "
                "same number (ticket 21)."
            )
        if len(wheels) > 1:
            return (
                f"{len(wheels)} wheels were gathered for {name}: "
                f"{', '.join(path.name for path in wheels)}. Each "
                "distribution is one wheel; more than one is a stale "
                "artefact beside the fresh one, and which of them PyPI would "
                "serve is not something to guess at."
            )
        sdists = [path for path in files if path.suffix != WHEEL_SUFFIX]
        if sdists and name != META_DISTRIBUTION:
            return (
                f"{', '.join(path.name for path in sdists)} would publish a "
                f"source distribution of {name}. Only the meta-package has "
                "one: the variants are a superbuild of PETSc, SLEPc, ADIOS2 "
                "and DOLFINx, so an sdist on the index is an install that "
                "tries to compile all of it on the user's machine."
            )
    return None


def gathered_problem(
    paths: Iterable[Path],
    version: str = __version__,
    distributions: Sequence[str] = DISTRIBUTIONS,
) -> str | None:
    """Report a directory that is not this release.

    Args:
        paths: Everything gathered for the upload.
        version: The version this release is, from the package.
        distributions: What a release publishes.

    Returns:
        A message, or ``None`` when the files are exactly this release.
    """
    gathered, problem = read_gathered(paths, distributions)
    return (
        problem
        or version_problem(gathered, version)
        or completeness_problem(gathered, distributions)
    )


def build_meta(
    outdir: Path, *, project: Path = META_PROJECT, python: str = sys.executable
) -> None:
    """Build the meta-package's wheel and sdist into a directory.

    Args:
        outdir: Where to leave them.
        project: The meta-package's project directory.
        python: Interpreter to run ``build`` under.

    Raises:
        subprocess.CalledProcessError: When the build fails.
    """
    check_call([python, "-m", "build", "--outdir", str(outdir), str(project)])


def collect_wheels(wheelhouse: Path, outdir: Path) -> list[Path]:
    """Copy the built wheels into the directory the upload reads.

    Every file is copied rather than the ones whose names look right: what
    the wheelhouse holds is what :func:`gathered_problem` has to be shown,
    and a filter here would hide the artefact it exists to catch.

    Args:
        wheelhouse: Directory the wheel jobs' artefacts were merged into.
        outdir: Where the upload reads from.

    Returns:
        The files that were copied.

    Raises:
        FileNotFoundError: When the wheelhouse does not exist. The artefacts
            are downloaded by the step before this one, so an empty path
            means that download did not happen rather than that this release
            has no wheels.
    """
    if not wheelhouse.is_dir():
        raise FileNotFoundError(
            f"no wheelhouse at {wheelhouse}. The variant wheels are the wheel "
            "jobs' artefacts, downloaded into it before this runs."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(wheelhouse.iterdir()):
        if path.is_file():
            copied.append(Path(shutil.copy2(path, outdir / path.name)))
    return copied


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for gathering a release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        required=True,
        help="directory the wheel jobs' artefacts were merged into",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="directory the upload reads; everything in it is published",
    )
    args = parser.parse_args(argv)

    try:
        collect_wheels(args.wheelhouse, args.outdir)
        build_meta(args.outdir)
    except FileNotFoundError as missing:
        print(f"ERROR: {missing}", file=sys.stderr)
        return 1

    gathered = sorted(path for path in args.outdir.iterdir() if path.is_file())
    problem = gathered_problem(gathered)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"==> publishing {__version__} to {args.index} ({INDEXES[args.index].upload})"
    )
    for path in gathered:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
