"""Solve something with a known answer, and a known complex answer.

The import checks prove the stack loads. They cannot prove it computes: a
wheel whose PETSc was configured for real scalars imports exactly as well as
one configured for complex, and so does a wheel whose MUMPS was silently left
out. Two problems with answers known in closed form are what separates them.

* **A Helmholtz problem whose exact solution is a plane wave.** ``exp(ikx)``
  solves ``-Δu - k²u = 0`` exactly, so imposing it on the boundary and
  solving gives a discrete solution whose distance from it is the
  discretisation error and nothing else. A real-scalar stack cannot even
  represent the boundary data, and a solve that quietly dropped the
  imaginary part would land nowhere near it (spec §11).
* **An eigenvalue problem whose spectrum is known.** The Dirichlet Laplacian
  on the unit square has eigenvalues ``π²(m² + n²)``, so the first three are
  ``2π²``, ``5π²``, ``5π²``. Reaching them goes through SLEPc, through a
  shift-and-invert spectral transform, and therefore through a sparse direct
  factorisation — which is the one part of the vendored stack no import can
  exercise.

Both run through ``dolfinx.fem.petsc``, whose module body loads libpetsc by
path (ticket 16), so importing this module is itself the check that the
solver layer of the payload is intact.

The tolerances are loose in absolute terms and tight in what they rule out:
they are perhaps ten times the error these discretisations actually produce,
which is far below the factor a wrong scalar type, a missing solver or a
mis-assembled operator would move the answer by.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wheeltest import payload

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Wave number of the Helmholtz problem: one wavelength across the domain,
#: which is enough oscillation for the phase to be wrong if the imaginary
#: part is mishandled, and few enough for a small mesh to resolve.
WAVE_NUMBER = 6.283185307179586

#: Cells per side, and the Lagrange degree solved with. Quadratics on this
#: mesh put the discretisation error near 1e-4, which leaves room for a
#: tolerance that is loose against mesh noise and tight against a broken
#: solve.
MESH_SIZE = 24
DEGREE = 2

#: How far the discrete solution may sit from the plane wave, relative to
#: the plane wave's own norm.
L2_TOLERANCE = 1e-3

#: The first three eigenvalues of the Dirichlet Laplacian on the unit
#: square, ``π²(m² + n²)`` for ``(1,1)``, ``(1,2)`` and ``(2,1)``.
EXPECTED_EIGENVALUES = (19.739208802178716, 49.34802200544679, 49.34802200544679)

#: How far a computed eigenvalue may sit from the exact one, relative. P1 on
#: this mesh overestimates them by about half a percent, from above, which is
#: what a Galerkin discretisation of this problem does.
EIGENVALUE_TOLERANCE = 0.03

#: How far an eigenvalue of a self-adjoint problem may leave the real axis,
#: relative to its own magnitude. Not zero, because the arithmetic is complex
#: and the answer is assembled from complex-valued inner products; nowhere
#: near the tolerance above, because the operator is Hermitian.
IMAGINARY_TOLERANCE = 1e-8

#: Where the eigensolver is aimed. Below the first eigenvalue, so
#: shift-and-invert converges onto the bottom of the spectrum rather than
#: onto whichever eigenvalue happens to be nearest a guess in the middle.
EIGENVALUE_TARGET = 10.0

#: What the Dirichlet rows are given in each operator, so the constrained
#: degrees of freedom contribute the eigenvalue ``1e6 / 1e-6`` rather than
#: ``1`` — far away from the part of the spectrum being asked for, instead of
#: in the middle of it.
STIFFNESS_DIAGONAL = 1e6
MASS_DIAGONAL = 1e-6


def error_problem(relative_error: float, tolerance: float = L2_TOLERANCE) -> str | None:
    """Report a Helmholtz solution that is not the plane wave it should be.

    Args:
        relative_error: The discrete solution's L2 distance from the exact
            one, over the exact one's L2 norm.
        tolerance: How far it may be.

    Returns:
        A message, or ``None`` when the solve landed on the answer.
    """
    if relative_error <= tolerance:
        return None
    return (
        f"the Helmholtz solve is {relative_error:.3e} away from the plane "
        f"wave that solves it exactly, relative, against a tolerance of "
        f"{tolerance:.0e}. The mesh and the element degree are fixed, so "
        "this is not discretisation error: the assembled operator, the "
        "linear solve or the complex arithmetic under them is wrong. Note "
        "that the wheel can import and report complex scalars and still "
        "fail here."
    )


def dtype_problem(dtype: Any) -> str | None:
    """Report a solution vector that does not hold complex numbers.

    Args:
        dtype: ``dtype`` of the solution's array.

    Returns:
        A message, or ``None`` when it is complex.
    """
    if str(dtype).startswith("complex"):
        return None
    return (
        f"the solution vector has dtype {dtype}, not a complex one. This "
        "distribution's PETSc is built with complex scalars and DOLFINx "
        "bakes that type into every vector it hands back (spec §6); a real "
        "one here means the wheel was assembled from a real-scalar prefix."
    )


def eigenvalue_problem(
    computed: Sequence[complex],
    expected: Sequence[float] = EXPECTED_EIGENVALUES,
    tolerance: float = EIGENVALUE_TOLERANCE,
) -> str | None:
    """Report an eigensolve that did not find the Dirichlet Laplacian's spectrum.

    Args:
        computed: The eigenvalues SLEPc converged on, in order.
        expected: The exact ones.
        tolerance: How far each may be, relative.

    Returns:
        A message naming the first one that is wrong, or ``None``.
    """
    if len(computed) < len(expected):
        return (
            f"SLEPc converged on {len(computed)} eigenvalues, and this check "
            f"asks for {len(expected)}. The spectral transform inverts the "
            "shifted operator with a sparse direct solve, so too few "
            "converged eigenvalues is usually the factorisation failing "
            "rather than the eigensolver."
        )
    for index, (found, exact) in enumerate(zip(computed, expected, strict=False)):
        if abs(found.imag) > IMAGINARY_TOLERANCE * abs(found):
            return (
                f"eigenvalue {index} is {found!r}, which is not real. The "
                "Dirichlet Laplacian is self-adjoint and its eigenvalues are "
                "real; an imaginary part this large means the operators were "
                "not assembled as the Hermitian pair the solve was told they "
                "are."
            )
        relative = abs(found.real - exact) / exact
        if relative > tolerance:
            return (
                f"eigenvalue {index} is {found.real:.6f}, and the Dirichlet "
                f"Laplacian's is {exact:.6f} — {relative:.2%} away, against "
                f"a tolerance of {tolerance:.0%}. The mesh is fixed, so this "
                "is the assembled operators or the eigensolver, not "
                "discretisation error."
            )
    return None


def helmholtz(comm: Any) -> tuple[float, Any]:
    """Solve the plane-wave Helmholtz problem and measure the solution.

    Args:
        comm: The communicator to solve on.

    Returns:
        The relative L2 error and the solution's ``dtype``.
    """
    import numpy as np
    import ufl
    from dolfinx import default_scalar_type, fem, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI

    domain = mesh.create_unit_square(comm, MESH_SIZE, MESH_SIZE)
    space = fem.functionspace(domain, ("Lagrange", DEGREE))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    coordinates = ufl.SpatialCoordinate(domain)
    exact = ufl.exp(1j * WAVE_NUMBER * coordinates[0])

    bilinear = (
        ufl.inner(ufl.grad(trial), ufl.grad(test))
        - WAVE_NUMBER**2 * ufl.inner(trial, test)
    ) * ufl.dx
    linear = ufl.inner(fem.Constant(domain, default_scalar_type(0)), test) * ufl.dx

    boundary_value = fem.Function(space)
    boundary_value.interpolate(lambda x: np.exp(1j * WAVE_NUMBER * x[0]))
    domain.topology.create_connectivity(1, 2)
    facets = mesh.exterior_facet_indices(domain.topology)
    condition = fem.dirichletbc(
        boundary_value, fem.locate_dofs_topological(space, 1, facets)
    )

    # A direct solve, so the check is on the operator and the answer rather
    # than on how many iterations a preconditioner needed. It is also what
    # puts MUMPS — one of the vendored solvers the licensing table pays for
    # (spec §7) — on the path of this test.
    problem = LinearProblem(
        bilinear,
        linear,
        bcs=[condition],
        petsc_options_prefix="wheeltest_smoke_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    solution = problem.solve()

    difference = solution - exact
    error = np.sqrt(
        domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(difference, difference) * ufl.dx)),
            MPI.SUM,
        )
    )
    reference = np.sqrt(
        domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(exact, exact) * ufl.dx)), MPI.SUM
        )
    )
    return abs(error / reference), solution.x.array.dtype


def eigenvalues(comm: Any, count: int = len(EXPECTED_EIGENVALUES)) -> list[complex]:
    """Find the bottom of the Dirichlet Laplacian's spectrum with SLEPc.

    Args:
        comm: The communicator to solve on.
        count: How many eigenvalues to ask for.

    Returns:
        The converged eigenvalues, smallest first.
    """
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import assemble_matrix
    from slepc4py import SLEPc

    domain = mesh.create_unit_square(comm, MESH_SIZE, MESH_SIZE)
    space = fem.functionspace(domain, ("Lagrange", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    domain.topology.create_connectivity(1, 2)
    facets = mesh.exterior_facet_indices(domain.topology)
    condition = fem.dirichletbc(
        fem.Function(space), fem.locate_dofs_topological(space, 1, facets)
    )

    stiffness = assemble_matrix(
        fem.form(ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx),
        bcs=[condition],
        diag=STIFFNESS_DIAGONAL,
    )
    stiffness.assemble()
    mass = assemble_matrix(
        fem.form(ufl.inner(trial, test) * ufl.dx),
        bcs=[condition],
        diag=MASS_DIAGONAL,
    )
    mass.assemble()

    solver = SLEPc.EPS().create(domain.comm)
    solver.setOperators(stiffness, mass)
    solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    solver.setTarget(EIGENVALUE_TARGET)
    solver.getST().setType(SLEPc.ST.Type.SINVERT)
    solver.setDimensions(count)
    solver.solve()
    return [solver.getEigenvalue(index) for index in range(solver.getConverged())]


def check(site: Path | None = None) -> str | None:
    """Solve both problems and report the first answer that is wrong.

    Args:
        site: Where the payload is installed. Used only to say where a
            failing import came from.

    Returns:
        A message, or ``None`` when the wheel computes what it should.
    """
    payload.use_installed()
    import dolfinx_solver

    try:
        from dolfinx.fem.petsc import LinearProblem  # noqa: F401
    except (ImportError, RuntimeError) as error:
        return (
            f"importing dolfinx.fem.petsc out of {site} failed: {error}. Its "
            "module body resolves libpetsc by path, so this is the payload's "
            "whole solver layer — LinearProblem, assemble_matrix, boundary "
            "conditions — and not one optional corner of it (ticket 16)."
        )

    # The communicator dolfinx_solver loaded, not a second import of
    # mpi4py: the ordering is the mechanism (spec §5, ADR-0002).
    comm = dolfinx_solver.mpi_module().COMM_WORLD

    relative_error, dtype = helmholtz(comm)
    problem = dtype_problem(dtype)
    if problem is None:
        problem = error_problem(relative_error)
    if problem is None:
        problem = eigenvalue_problem(eigenvalues(comm))
    return problem


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the numerical smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args(argv)

    problem = check(args.site)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"complex Helmholtz solved to within {L2_TOLERANCE:.0e} of the plane "
        f"wave, and SLEPc found the Dirichlet Laplacian's first "
        f"{len(EXPECTED_EIGENVALUES)} eigenvalues"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
