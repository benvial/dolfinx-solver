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
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


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


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the prefix layout fix-up."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    library_dir = unify_lib_directories(args.prefix)
    print(f"lib64 aliases {library_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
