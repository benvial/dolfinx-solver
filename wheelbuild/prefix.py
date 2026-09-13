"""Keep one library directory in the shared install prefix.

Everything the container builds — the Fortran half of MPICH, PETSc and its
``--download-*`` tree, SLEPc, ADIOS2, KaHIP, DOLFINx — installs into one
prefix, and the halves disagree about where a library goes. CMake's
``GNUInstallDirs`` picks ``lib64`` on the RHEL-family system the
manylinux_2_34 image is built from, while MPICH's own build system and PETSc's
configure use ``lib``. A consumer that looks in the wrong one fails to link
against a library that is sitting in the other.

Making ``lib64`` a symlink to ``lib`` before anything is built means both
spellings resolve to the same files whichever convention a sub-project picks,
and the wheel assembly step then has one directory to vendor from.

The prefix also belongs to exactly one scalar variant. ``PetscScalar`` is
baked into libpetsc, the bindings and DOLFINx's binaries alike, so
``dolfinx-solver-complex`` and ``dolfinx-solver-real`` are two wheels built
from two prefixes (spec §6) — and because cache stamps are never pruned,
building the second variant into the first one's prefix would leave the first
one's stamps behind and take the cached path over a stack of the wrong scalar
type. This module records which variant claimed the prefix and refuses the
other, which is the one failure an operator can fix with a second
``BUILD_ROOT``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from wheelbuild import markers
from wheelbuild.petsc import SCALAR_TYPE, SCALAR_TYPES

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Where the prefix records the variant it was built for.
VARIANT_FILE = ".scalar-type"

#: The PETSc stage's cache stamp, which carried the scalar type before this
#: marker existed: ``.petsc-<release>-<scalar>.installed`` (ticket 12). It is
#: what a warm cache from before the marker is read by, so a prefix built for
#: one variant is still refused to the other rather than adopted and then
#: half-overwritten.
PETSC_STAMP_GLOB = ".petsc-*.installed"


def unify_lib_directories(prefix: Path) -> Path:
    """Make ``<prefix>/lib64`` an alias of ``<prefix>/lib``.

    Anything already installed into a real ``lib64`` directory is moved into
    ``lib`` first, so this is safe to run against a warm build cache — which
    the container entry point does on every start, cached tree or not.

    Args:
        prefix: The shared install prefix. Created if it does not exist.

    Returns:
        The ``lib`` directory every library now lives in.
    """
    library_dir = prefix / "lib"
    library_dir.mkdir(parents=True, exist_ok=True)
    alias = prefix / "lib64"

    if alias.is_symlink():
        alias.unlink()
    elif alias.is_dir():
        _merge_into(alias, library_dir)
        shutil.rmtree(alias)

    alias.symlink_to("lib", target_is_directory=True)
    return library_dir


def _merge_into(source: Path, destination: Path) -> None:
    """Move everything from ``source`` into ``destination``, merging directories.

    Directories present on both sides are merged recursively — replacing one
    wholesale, or skipping it, would drop the files only the other side has
    (the CMake package configuration directories, for instance). A file that
    already exists in ``destination`` wins, since that is the copy the build
    has been linking against all along.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        target = destination / path.name
        if path.is_dir() and not path.is_symlink():
            _merge_into(path, target)
        elif not target.exists() and not target.is_symlink():
            shutil.move(str(path), str(target))


def claim_variant(prefix: Path, scalar_type: str = SCALAR_TYPE) -> Path:
    """Record which scalar variant owns a prefix, and refuse the other one.

    Args:
        prefix: The shared install prefix. Created if it does not exist.
        scalar_type: The variant about to build into it, ``complex`` or
            ``real``.

    Returns:
        The prefix.

    Raises:
        ValueError: When the prefix was already built for the other variant.
        OSError: When the marker cannot be written.
    """
    if scalar_type not in SCALAR_TYPES:
        raise ValueError(
            f"{scalar_type!r} is not a scalar variant this build knows: "
            f"{', '.join(SCALAR_TYPES)} (spec §6)."
        )
    prefix.mkdir(parents=True, exist_ok=True)
    marked = markers.text(prefix / VARIANT_FILE)
    claimed = _claimed_variant(prefix, marked)
    if claimed is not None and claimed != scalar_type:
        raise ValueError(
            f"{prefix} was built for the {claimed}-scalar variant and "
            f"this build is the {scalar_type}-scalar one. The two are "
            "separate wheels because PetscScalar is baked into every "
            "binary in the prefix (spec §6), and the stages here decide "
            "'already built' on stamps that are never pruned — so "
            f"building {scalar_type} in here would re-check the "
            f"{claimed} stack rather than replace it. Point BUILD_ROOT at "
            "a second directory, one per variant."
        )
    # The warm path does not write at all: the claim is already there, and a
    # rewrite is a window ticket 24 has no reason to open.
    if marked != scalar_type:
        markers.write(prefix / VARIANT_FILE, scalar_type)
    return prefix


def _claimed_variant(prefix: Path, marked: str | None) -> str | None:
    """Return the variant a prefix already belongs to, or ``None`` if it is new.

    Args:
        prefix: The prefix to judge.
        marked: What its variant marker says, as
            :func:`wheelbuild.markers.text` reports it. Read by the caller,
            which also needs to know whether the marker is already the claim
            being made.

    The marker is the answer once it exists and says something this build has
    a name for. Before it did, the PETSc stage's cache stamp was the only
    thing in the prefix naming a scalar type, and a warm cache restored from
    CI or left on a workstation has one — so an unmarked prefix is read from
    its stamps rather than adopted blind, which is what keeps the one
    migration the marker exists for from overwriting a complex stack with a
    real one.

    A marker whose text is not one of :data:`SCALAR_TYPES` is treated as no
    marker at all. Markers are replaced rather than truncated
    (:mod:`wheelbuild.markers`), so a killed run is no longer how an empty one
    appears, but a truncated or hand-edited file still is — and obeying that
    would refuse every build of *either*
    variant in the name of a variant that does not exist, with nothing to
    offer but deleting a prefix that holds hours of superbuild. The stamps are
    the better evidence anyway: they are written by the stage that did the
    work (ticket 12).
    """
    if marked in SCALAR_TYPES:
        return marked
    for stamp in sorted(prefix.glob(PETSC_STAMP_GLOB)):
        candidate = stamp.name.removesuffix(".installed").rsplit("-", 1)[-1]
        if candidate in SCALAR_TYPES:
            return candidate
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the prefix layout fix-up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument(
        "--scalar-type",
        default=SCALAR_TYPE,
        choices=SCALAR_TYPES,
        help="the variant this prefix belongs to (spec §6)",
    )
    args = parser.parse_args(argv)
    # Claimed before anything is laid out: a prefix the other variant owns is
    # one this build must not touch at all, not one it merges lib64 into first.
    try:
        claim_variant(args.prefix, args.scalar_type)
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    library_dir = unify_lib_directories(args.prefix)
    print(f"lib64 aliases {library_dir}; prefix is the {args.scalar_type} variant's")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
