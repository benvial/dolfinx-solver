"""Import-order and environment guards that run before any compiled module.

DOLFINx, mpi4py and petsc4py all live in one process and must agree on one
MPI runtime and one PETSc. Two things decide whether they do, and both are
settled at import time, before a single binding is loaded:

* ``mpi4py.MPI`` must be imported first. Its ``_mpiabi`` loader dlopens
  ``libmpi.so.12`` out of the environment's prefix (the PyPI ``mpich`` wheel),
  and every ``DT_NEEDED`` binding afterwards resolves to that same copy. Let a
  vendored library load first and the process ends up with two MPI runtimes,
  which shows up as a hang or a wrong answer rather than an error.
* No foreign ``petsc4py`` may shadow the one inside this wheel. Ours is built
  against the vendored PETSc; another one — from the system, from a
  ``pip install petsc4py`` — is a second PETSc in the process with a different
  scalar type and a different libmpi. There is no way to make that work, so
  it is refused with a message that says what to uninstall.

That second check has two halves, because a foreign petsc4py arrives two ways.
Installing the PyPI project puts its files exactly where ours live and leaves
the import path looking untouched, so what gives it away is its installed
distribution; one that merely shadows ours from elsewhere on ``sys.path``
leaves no distribution here at all, so what gives that one away is where the
import resolves to.

Which of the two wheels this is has to be discovered rather than declared.
``dolfinx-solver-complex`` and ``dolfinx-solver-real`` ship the same payload
compiled against a different ``PetscScalar`` (spec §6), and the variant is
declared in ``wheelbuild.petsc.SCALAR_TYPE`` — which is a build driver, not
shipped inside the wheel. Nor can the payload ask petsc4py: this module exists
to run *before* petsc4py is imported. What is left is the payload's own
installed distribution, which is the true answer anyway — it is the wheel that
was unpacked here — and it is the name a message has to use when it tells
someone to reinstall. Where no such distribution is installed, as in a source
checkout, the messages name no variant rather than guessing one: telling a
real install to reinstall ``dolfinx-solver-complex`` would put a complex stack
over its real one.

Every check takes its inputs as arguments, so the ordering can be tested
without a working MPI stack.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

#: Directory this package is installed in.
PACKAGE_DIR = Path(__file__).resolve().parent

#: Import name of the PETSc bindings this wheel ships beside itself.
PETSC4PY = "petsc4py"

#: Import name that must reach the loader first.
MPI_MODULE = "mpi4py.MPI"

#: PyPI projects that install a PETSc or SLEPc of their own. This wheel ships
#: all four under their upstream import names with no distribution of their
#: own, so any of these being installed means a second stack is present.
CONFLICTING_DISTRIBUTIONS = frozenset({"petsc", "petsc4py", "slepc", "slepc4py"})

#: Name shared by every distribution that ships this payload. The bare name is
#: the meta-package, which depends on a variant rather than being one.
DISTRIBUTION_STEM = "dolfinx-solver"

#: The scalar variants there is a wheel for (spec §6), in the spelling the
#: distribution name carries. ``wheelbuild.petsc.SCALAR_TYPES`` is the build
#: side of the same pair; the two meet in the distribution name the assembler
#: writes and the installer records.
SCALAR_VARIANTS = ("complex", "real")

#: The distributions those variants are published under.
VARIANT_DISTRIBUTIONS = frozenset(
    f"{DISTRIBUTION_STEM}-{variant}" for variant in SCALAR_VARIANTS
)


def canonical_name(name: str) -> str:
    """Return a distribution name in PEP 503 normalised form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_distribution_names() -> list[str]:
    """Return the normalised names of every installed distribution."""
    return [
        canonical_name(name)
        for distribution in importlib.metadata.distributions()
        if (name := distribution.metadata["Name"]) is not None
    ]


def shipping_distribution(names: Iterable[str]) -> str | None:
    """Return the variant distribution this payload was installed from.

    Args:
        names: Normalised names of the installed distributions, as
            :func:`installed_distribution_names` reports them.

    Returns:
        ``dolfinx-solver-complex`` or ``dolfinx-solver-real``, or ``None``
        when neither is installed — a source checkout, a wheel unpacked by
        hand — or when both are, which is an environment whose payload cannot
        be attributed to either.
    """
    installed = sorted(VARIANT_DISTRIBUTIONS.intersection(names))
    if len(installed) != 1:
        return None
    return installed[0]


def scalar_variant(distribution: str | None) -> str | None:
    """Return the scalar type a variant distribution name carries.

    Args:
        distribution: A distribution name, as :func:`shipping_distribution`
            reports it, or ``None``.

    Returns:
        ``complex``, ``real``, or ``None`` when the name is not one of the
        variant distributions.
    """
    if distribution is None or distribution not in VARIANT_DISTRIBUTIONS:
        return None
    return distribution.removeprefix(f"{DISTRIBUTION_STEM}-")


def vendored_stack(distribution: str | None) -> str:
    """Describe the stack inside the wheel, naming its scalar type if known.

    Args:
        distribution: The variant distribution this payload was installed
            from, or ``None`` when it could not be determined.

    Returns:
        A noun phrase for the four vendored projects, qualified by the scalar
        variant only when the payload can actually tell which one it is.
    """
    variant = scalar_variant(distribution)
    qualifier = f"{variant}-scalar " if variant is not None else ""
    return f"its own {qualifier}PETSc, SLEPc, petsc4py and slepc4py"


def conflicting_distribution_problem(names: Iterable[str]) -> str | None:
    """Report PETSc or SLEPc distributions installed beside this wheel.

    Args:
        names: Normalised names of the installed distributions, as
            :func:`installed_distribution_names` reports them.

    Returns:
        A message naming what to uninstall, or ``None`` when none of them is
        installed.
    """
    names = list(names)
    conflicting = sorted(CONFLICTING_DISTRIBUTIONS.intersection(names))
    if not conflicting:
        return None
    distribution = shipping_distribution(names)
    subject = distribution if distribution is not None else "This distribution"
    remedy = (
        f"reinstall {distribution}"
        if distribution is not None
        else f"reinstall the {DISTRIBUTION_STEM} distribution you installed"
    )
    return (
        f"{', '.join(conflicting)} {'is' if len(conflicting) == 1 else 'are'} "
        f"installed in this environment. {subject} ships "
        f"{vendored_stack(distribution)} inside the wheel, "
        "and the PyPI projects of those names overwrite them with a build "
        "against a different PETSc and a different MPI, which cannot be made "
        f"to work in one process. Uninstall {' '.join(conflicting)}, then "
        f"{remedy} to restore the files it shipped."
    )


def petsc4py_origin() -> Path | None:
    """Return where ``petsc4py`` would be imported from, without importing it.

    Returns:
        The path recorded on the module spec, or ``None`` when no petsc4py is
        importable at all.
    """
    try:
        spec = importlib.util.find_spec(PETSC4PY)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def foreign_petsc4py_problem(
    origin: Path | None,
    package_dir: Path = PACKAGE_DIR,
    distribution: str | None = None,
) -> str | None:
    """Report a petsc4py that is not the one installed beside this package.

    Args:
        origin: Path petsc4py resolves to, as :func:`petsc4py_origin` reports
            it, or ``None`` when petsc4py is not importable.
        package_dir: Directory this package is installed in. The wheel puts
            petsc4py next to it, so its parent is the directory both share.
        distribution: The variant distribution this payload was installed
            from, as :func:`shipping_distribution` reports it. Left ``None``,
            the message names no scalar variant.

    Returns:
        A message naming what to uninstall, or ``None`` when the resolvable
        petsc4py is ours or absent.
    """
    if origin is None:
        return None
    try:
        shipped_beside = origin.resolve().parent.parent == package_dir.parent
    except OSError:  # pragma: no cover - an unresolvable path is foreign too
        shipped_beside = False
    if shipped_beside:
        return None
    vendor = distribution if distribution is not None else "This wheel"
    return (
        f"petsc4py would be imported from {origin}, which is not the build "
        f"shipped inside this wheel ({package_dir.parent}). {vendor} vendors "
        f"{vendored_stack(distribution)}; "
        "a second PETSc in the same process has a different scalar type and a "
        "different MPI, and cannot be made to work. Uninstall the other "
        "petsc4py (and any petsc, slepc or slepc4py beside it), or install "
        f"{vendor.lower() if distribution is None else distribution} in an "
        "environment of its own."
    )


def bootstrap(
    import_module: Callable[[str], Any] = importlib.import_module,
    petsc4py_origin: Callable[[], Path | None] = petsc4py_origin,
    package_dir: Path = PACKAGE_DIR,
    installed_distribution_names: Callable[
        [], Iterable[str]
    ] = installed_distribution_names,
) -> Any:
    """Refuse a foreign petsc4py, then load MPI ahead of everything else.

    Args:
        import_module: How to import a module by name. Injected so the import
            order can be asserted without MPI present.
        petsc4py_origin: Where petsc4py would come from.
        package_dir: Directory this package is installed in.
        installed_distribution_names: Normalised names of the distributions
            installed in this environment.

    Returns:
        The imported ``mpi4py.MPI`` module.

    Raises:
        ImportError: When a foreign PETSc stack is installed or shadows the
            vendored one, or when mpi4py is missing.
    """
    names = list(installed_distribution_names())
    problem = conflicting_distribution_problem(names)
    if problem is None:
        problem = foreign_petsc4py_problem(
            petsc4py_origin(), package_dir, shipping_distribution(names)
        )
    if problem is not None:
        raise ImportError(problem)
    try:
        return import_module(MPI_MODULE)
    except ImportError as error:
        raise ImportError(
            f"dolfinx-solver needs mpi4py imported before its compiled "
            f"modules, and importing {MPI_MODULE} failed: {error}. Install "
            "mpi4py and the PyPI mpich wheel (pip install mpich mpi4py); an "
            "Open MPI build will not do, this wheel is MPICH-ABI only."
        ) from error
