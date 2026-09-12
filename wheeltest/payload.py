"""Import the installed payload, not the repository it was built from.

The stages are run as modules of this repository inside the venv the wheel
was installed into, which means the repository is on ``sys.path`` ahead of
that venv's ``site-packages``. The repository also contains a directory
called ``dolfinx_solver``: the source of the package the wheel ships. Left
alone, every stage would import the source copy — whose ``_bootstrap``
computes ``PACKAGE_DIR`` from its own location, decides the installed
petsc4py beside the wheel is a foreign one, and refuses the import.

The failure is loud, which is lucky, because the quiet version of it is
worse: a suite that imports the repository's ``dolfinx_solver`` and the
venv's ``dolfinx`` has proven nothing about either.

So the repository comes off the path before the payload goes on it. It is
already imported by then — that is what makes this callable at all — and
nothing a stage imports afterwards comes from anywhere but the install
under test.

This module deliberately imports nothing of its own, so a stage can call it
before it has decided what else to import.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The repository, which is what has to leave the path.
REPO_ROOT = Path(__file__).resolve().parent.parent


def repository_entry(entry: str, repo_root: Path = REPO_ROOT) -> bool:
    """Report whether a ``sys.path`` entry points at the repository.

    Args:
        entry: One entry, as ``sys.path`` holds it. The empty string is the
            working directory, which is the repository whenever the suite is
            run from a checkout — the usual case.
        repo_root: The repository root.

    Returns:
        Whether it resolves to the repository.
    """
    try:
        return Path(entry or ".").resolve() == repo_root
    except OSError:  # pragma: no cover - an unresolvable entry is not ours
        return False


def installed_only(path: list[str], repo_root: Path = REPO_ROOT) -> list[str]:
    """Return a module path with the repository's own copies taken off it.

    Args:
        path: The path to filter.
        repo_root: The repository root.

    Returns:
        The entries that are not the repository.
    """
    return [entry for entry in path if not repository_entry(entry, repo_root)]


def use_installed() -> list[str]:
    """Take the repository off this interpreter's module path.

    Returns:
        The entries that were removed, so a stage can say so.
    """
    removed = [entry for entry in sys.path if repository_entry(entry)]
    sys.path[:] = installed_only(sys.path)
    return removed
