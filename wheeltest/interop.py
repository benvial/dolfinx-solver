"""Two ranks, three libraries, one communicator.

Run under ``mpiexec -n 2``. Everything else in the suite runs one rank in
one process, where MPI is initialised and never used; the claim spec §11
makes is about a process where mpi4py, petsc4py and DOLFINx each hold a
handle on the same MPI and each drive collective operations through it.
Three things could be wrong there, and none of them is visible serially:

* **The ranks cannot see each other.** A launcher from another MPI — the
  machine's Open MPI ``mpirun``, say — starts two processes that each
  believe they are a one-rank job. Nothing errors; the run just does
  everything twice. ``COMM_WORLD.size`` is what says otherwise, so it is
  asserted before anything else.
* **The mesh is not really distributed.** The partitioner is the vendored
  PT-SCOTCH, reached through DOLFINx's graph layer, and a mesh that failed
  to partition leaves one rank with every cell and the other with none —
  which still solves, still agrees, and proves nothing.
* **The libraries hold different communicators.** A PETSc vector built from
  a DOLFINx function has a communicator of its own; that its size matches
  and that its collective norm equals the one mpi4py computes by hand from
  the same local data is the shared-communicator claim, stated as an
  equation rather than as a diagram.

The ranks pool their findings before reporting, so a failure on rank 1 is
not a run that hangs waiting for it — and so the message a user sees names
the rank it came from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wheelbuild import import_check, mpich
from wheeltest import payload

if TYPE_CHECKING:
    from collections.abc import Sequence

#: How many ranks the interop stage is run with (spec §11).
RANKS = 2

#: Cells per side of the mesh being partitioned. Small enough to build in
#: under a second, large enough that a partitioner has something to divide.
MESH_SIZE = 16

#: The smallest share of the cells a rank may own for the mesh to count as
#: distributed. Well below an even split, because PT-SCOTCH balances a
#: triangulated square better than this and the check is for a partition
#: that did not happen at all.
MINIMUM_SHARE = 0.1

#: How far PETSc's collective norm may sit from the one computed from the
#: same numbers through mpi4py, relative. Both sum the same squares in some
#: order, so the only difference allowed is the order they were summed in.
NORM_TOLERANCE = 1e-12


def size_problem(size: int, expected: int = RANKS) -> str | None:
    """Report a job whose ranks are not in one communicator.

    Args:
        size: What ``MPI.COMM_WORLD`` reports its size as.
        expected: How many ranks the job was launched with.

    Returns:
        A message, or ``None`` when they agree.
    """
    if size == expected:
        return None
    return (
        f"MPI.COMM_WORLD has {size} rank(s) in a job launched with "
        f"{expected}. The launcher has to be the hydra the PyPI mpich wheel "
        "installs beside the interpreter: another MPI's mpiexec starts "
        "processes that each form a communicator of one, which runs the "
        "whole test twice and passes."
    )


def partition_problem(
    local_cells: Sequence[int], global_cells: int, minimum: float = MINIMUM_SHARE
) -> str | None:
    """Report a mesh that was not spread over the ranks.

    Args:
        local_cells: Cells each rank owns, by rank.
        global_cells: Cells in the mesh.
        minimum: The smallest share of them a rank may own.

    Returns:
        A message, or ``None`` when every rank got a share.
    """
    if sum(local_cells) != global_cells:
        return (
            f"the ranks own {sum(local_cells)} cells between them "
            f"({', '.join(str(count) for count in local_cells)}) and the "
            f"mesh has {global_cells}. Owned cells partition the mesh, so "
            "these have to agree; they do not when the ranks built "
            "different meshes, which is what two disjoint communicators "
            "look like from here."
        )
    starved = [
        rank for rank, count in enumerate(local_cells) if count < minimum * global_cells
    ]
    if starved:
        return (
            f"rank(s) {', '.join(str(rank) for rank in starved)} own less "
            f"than {minimum:.0%} of the mesh "
            f"({', '.join(str(count) for count in local_cells)} of "
            f"{global_cells} cells). The mesh is built with the vendored "
            "PT-SCOTCH partitioner (spec §11); a partition this lopsided "
            "means it did not run, and the work is not distributed at all."
        )
    return None


def norm_problem(
    petsc_norm: float, mpi4py_norm: float, tolerance: float = NORM_TOLERANCE
) -> str | None:
    """Report PETSc and mpi4py disagreeing about a vector they both reduced.

    Args:
        petsc_norm: What ``Vec.norm()`` returned — a collective over
            petsc4py's communicator.
        mpi4py_norm: The same norm, summed over mpi4py's.
        tolerance: How far apart they may be, relative.

    Returns:
        A message, or ``None`` when the two reductions agree.
    """
    scale = max(abs(petsc_norm), abs(mpi4py_norm), 1.0)
    if abs(petsc_norm - mpi4py_norm) <= tolerance * scale:
        return None
    return (
        f"PETSc reduced the vector to {petsc_norm!r} and mpi4py reduced the "
        f"same local values to {mpi4py_norm!r}. Both are collectives over "
        "what is supposed to be one communicator holding the same ranks; "
        "two answers means the vector's communicator is not the one mpi4py "
        "is reducing over, and every parallel result out of this wheel is "
        "then whatever the two happened to produce (spec §5, ADR-0002)."
    )


def observe(comm: Any) -> dict[str, Any]:
    """Build a distributed mesh and measure what the three libraries say.

    Args:
        comm: ``MPI.COMM_WORLD``.

    Returns:
        The facts the judgements above are made against.
    """
    import numpy as np
    from dolfinx import fem, graph, mesh
    from mpi4py import MPI

    partitioner = mesh.create_cell_partitioner(
        graph.partitioner_scotch(), mesh.GhostMode.shared_facet, 1
    )
    domain = mesh.create_unit_square(
        comm, MESH_SIZE, MESH_SIZE, partitioner=partitioner
    )
    cell_map = domain.topology.index_map(domain.topology.dim)

    space = fem.functionspace(domain, ("Lagrange", 1))
    function = fem.Function(space)
    # Distinct values per rank, so a norm computed from one rank's data
    # alone is a different number rather than a coincidence.
    function.x.array[:] = comm.rank + 1
    vector = function.x.petsc_vec

    owned = function.x.array[: space.dofmap.index_map.size_local]
    local_square = float(np.sum(np.abs(owned) ** 2))

    return {
        "size": comm.size,
        "local_cells": comm.allgather(cell_map.size_local),
        "global_cells": cell_map.size_global,
        "petsc_norm": vector.norm(),
        "mpi4py_norm": float(np.sqrt(comm.allreduce(local_square, MPI.SUM))),
        "petsc_comm_size": vector.getComm().getSize(),
    }


def facts_problem(facts: dict[str, Any]) -> str | None:
    """Report the first thing wrong with what the ranks measured.

    Args:
        facts: What :func:`observe` returned.

    Returns:
        A message, or ``None``.
    """
    problem = size_problem(facts["size"])
    if problem is not None:
        return problem
    if facts["petsc_comm_size"] != facts["size"]:
        return (
            f"a PETSc vector built from a DOLFINx function has "
            f"{facts['petsc_comm_size']} rank(s) on its communicator, and "
            f"MPI.COMM_WORLD has {facts['size']}. DOLFINx passes its own "
            "communicator to PETSc when it makes the vector, so a different "
            "size means the two libraries are not holding the same MPI."
        )
    problem = partition_problem(facts["local_cells"], facts["global_cells"])
    if problem is not None:
        return problem
    return norm_problem(facts["petsc_norm"], facts["mpi4py_norm"])


def check(site: Path) -> str | None:
    """Run the parallel checks and report the first problem on any rank.

    Args:
        site: Where the payload is installed. Reported with a failure.

    Returns:
        A message naming the rank it came from, on every rank, or ``None``.
    """
    payload.use_installed()
    import dolfinx_solver

    # The package's own accessor, rather than importing mpi4py again: it
    # hands back the module dolfinx_solver loaded before anything compiled,
    # which is the one the whole linkage is about (spec §5).
    comm = dolfinx_solver.mpi_module().COMM_WORLD

    # Every rank asks the same question at the same point and every verdict
    # is agreed before the next one is asked. A rank that returned early on
    # its own finding would leave the others inside a collective that never
    # completes, and a suite that hangs says nothing about what went wrong —
    # which is the failure this ordering exists to avoid, and the reason
    # observe() is not called until the ranks agree they should.
    problem = _agreed(
        comm,
        import_check.mpi_problem(import_check.MAPS_PATH.read_text(encoding="utf-8")),
        site,
    )
    if problem is not None:
        return problem
    return _agreed(comm, facts_problem(observe(comm)), site)


def _agreed(comm: Any, problem: str | None, site: Path) -> str | None:
    """Return the first problem any rank found, on every rank.

    Args:
        comm: The communicator every rank is in.
        problem: What this rank found, or ``None``.
        site: Where the payload is installed.

    Returns:
        The message, named by the rank it came from, or ``None`` when no
        rank found anything.
    """
    for rank, message in enumerate(comm.allgather(problem)):
        if message is not None:
            return f"on rank {rank} of {comm.size} (payload in {site}): {message}"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the parallel interop test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args(argv)

    problem = check(args.site)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    from mpi4py import MPI

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"{MPI.COMM_WORLD.size} ranks share one {mpich.MPI_SONAME}, one "
            "communicator and one PT-SCOTCH-partitioned mesh; PETSc and "
            "mpi4py reduce it to the same norm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
