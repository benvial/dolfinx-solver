"""DOLFINx with a PETSc of one scalar type, packaged as a binary wheel.

Which scalar type is the distribution's, not this module's:
``dolfinx-solver-complex`` and ``dolfinx-solver-real`` install the same
package compiled against a different ``PetscScalar`` (spec §6), and
:func:`._bootstrap.shipping_distribution` is what reads back which of the two
was installed here.

Importing this package is what makes the wheel's stack safe to use: it orders
``mpi4py.MPI`` ahead of every compiled module and refuses an environment where
a foreign petsc4py would shadow the vendored one (see :mod:`._bootstrap`).
The library itself is then imported under its upstream names — ``dolfinx``,
``petsc4py``, ``slepc4py`` — exactly as a conda or source install would, so
downstream code never mentions ``dolfinx_solver`` except to import it first::

    import dolfinx_solver  # noqa: F401 - orders MPI ahead of the bindings
    import dolfinx

This package carries no DOLFINx API of its own.
"""

from __future__ import annotations

from typing import Any

from dolfinx_solver._bootstrap import bootstrap
from dolfinx_solver._version import DOLFINX_VERSION, RELEASE, __version__

__all__ = ["DOLFINX_VERSION", "RELEASE", "__version__", "mpi_module"]

_MPI = bootstrap()


def mpi_module() -> Any:
    """Return the ``mpi4py.MPI`` module this package loaded first.

    Returns:
        The module object, so callers can assert the ordering actually
        happened rather than trusting that it did.
    """
    return _MPI
