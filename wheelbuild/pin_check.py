"""Keep the declared ``mpich`` bound and the MPICH this build vendors in step.

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
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from wheelbuild.mpich import MPICH_VERSION, series

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The distribution the runtime ``libmpi`` comes from.
MPICH_DISTRIBUTION = "mpich"

#: This project's metadata, where the pin is declared.
PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


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
    dependencies = (
        tomllib.loads(pyproject_text).get("project", {}).get("dependencies", [])
    )
    for dependency in dependencies:
        # A requirement may spell the project any way PEP 503 normalises to
        # the same name, so the comparison is on the canonical form.
        if canonicalize_name(Requirement(dependency).name) == MPICH_DISTRIBUTION:
            return str(dependency)
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
    """Command-line entry point for the pin check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT_PATH)
    args = parser.parse_args(argv)

    try:
        declared = declared_mpich_requirement(
            args.pyproject.read_text(encoding="utf-8")
        )
    except LookupError as undeclared:
        print(f"ERROR: {undeclared}", file=sys.stderr)
        return 1

    problem = pin_problem(declared, MPICH_VERSION)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    print(f"the declared {declared!r} names the MPICH {MPICH_VERSION} series")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
