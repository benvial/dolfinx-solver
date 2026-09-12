"""Solve something with a known answer, in the scalars this wheel was built for.

The import checks prove the stack loads. They cannot prove it computes: a
wheel whose PETSc was configured for real scalars imports exactly as well as
one configured for complex, and so does a wheel whose MUMPS was silently left
out. Two problems with answers known in closed form are what separates them.

* **A boundary-value problem whose exact solution is written down.** Which one
  it is follows :data:`wheelbuild.petsc.SCALAR_TYPE`, because the variant is
  baked into every vector DOLFINx hands back and a real-scalar stack cannot
  even represent complex boundary data (spec §6, §11):

  * ``complex`` — Helmholtz, ``-Δu - k²u = 0``, whose plane wave ``exp(ikx)``
    solves it exactly. Imposing it on the boundary and solving gives a
    discrete solution whose distance from it is the discretisation error and
    nothing else, and a solve that quietly dropped the imaginary part would
    land nowhere near it.
  * ``real`` — Poisson, ``-Δu = 2π² sin(πx) sin(πy)``, whose exact solution
    ``sin(πx) sin(πy)`` vanishes on the boundary of the unit square. The same
    measurement, in the scalars that variant has.

  Either way the answer is judged twice: on its distance from the exact
  solution, and on the ``dtype`` it came back in, which is the variant read
  off the answer rather than off a build flag.

* **An eigenvalue problem whose spectrum is known.** The Dirichlet Laplacian
  on the unit square has eigenvalues ``π²(m² + n²)``, so the first three are
  ``2π²``, ``5π²``, ``5π²``. Reaching them goes through SLEPc, through a
  shift-and-invert spectral transform, and therefore through a sparse direct
  factorisation — which is the one part of the vendored stack no import can
  exercise. It is the same problem for both variants: the operators are real
  and symmetric, and the spectrum is real.

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
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from wheelbuild import petsc
from wheeltest import payload

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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

#: How far the discrete solution may sit from the exact one, relative to the
#: exact one's own norm.
L2_TOLERANCE = 1e-3

#: What the linear solve is told to do. A direct solve, so the check is on
#: the operator and the answer rather than on how many iterations a
#: preconditioner needed. It is also what puts MUMPS — one of the vendored
#: solvers the licensing table pays for (spec §7) — on the path of this test.
DIRECT_SOLVE = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

#: The first three eigenvalues of the Dirichlet Laplacian on the unit
#: square, ``π²(m² + n²)`` for ``(1,1)``, ``(1,2)`` and ``(2,1)``.
EXPECTED_EIGENVALUES = (19.739208802178716, 49.34802200544679, 49.34802200544679)

#: How far a computed eigenvalue may sit from the exact one, relative. P1 on
#: this mesh overestimates them by about half a percent, from above, which is
#: what a Galerkin discretisation of this problem does.
EIGENVALUE_TOLERANCE = 0.03

#: How far an eigenvalue of a self-adjoint problem may leave the real axis,
#: relative to its own magnitude. Not zero, because under complex scalars the
#: answer is assembled from complex-valued inner products; nowhere near the
#: tolerance above, because the operator is Hermitian.
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


class Problem(NamedTuple):
    """The boundary-value problem one scalar variant is proved by.

    Attributes:
        name: What it is called in the messages a user reads.
        exact: The solution it is measured against, as a phrase that follows
            "solves onto".
        dtype_prefix: What ``numpy`` spells the start of the solution
            vector's ``dtype`` as, under this variant.
        solve: Solves it on a communicator and returns the relative L2 error
            and the solution's ``dtype``.
    """

    name: str
    exact: str
    dtype_prefix: str
    solve: Callable[[Any], tuple[float, Any]]


def _relative_error(domain: Any, discrete: Any, exact: Any) -> float:
    """Return the L2 distance between two fields, over the exact one's norm.

    Args:
        domain: The mesh both are defined on, whose communicator the
            integrals are reduced over.
        discrete: What was solved for.
        exact: What it should be.

    Returns:
        The relative error, as a non-negative real number.
    """
    import numpy as np
    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    def norm(field: Any) -> Any:
        form = fem.form(ufl.inner(field, field) * ufl.dx)
        return np.sqrt(domain.comm.allreduce(fem.assemble_scalar(form), MPI.SUM))

    return float(abs(norm(discrete - exact) / norm(exact)))


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
    condition = _dirichlet(domain, space, boundary_value)

    problem = LinearProblem(
        bilinear,
        linear,
        bcs=[condition],
        petsc_options_prefix="wheeltest_smoke_",
        petsc_options=dict(DIRECT_SOLVE),
    )
    solution = problem.solve()

    return _relative_error(domain, solution, exact), solution.x.array.dtype


def poisson(comm: Any) -> tuple[float, Any]:
    """Solve the sine-product Poisson problem and measure the solution.

    The source is chosen so the exact solution is ``sin(πx) sin(πy)``, which
    is zero on the boundary of the unit square: the boundary condition is the
    homogeneous one, and every scalar in the problem is real (spec §11).

    Args:
        comm: The communicator to solve on.

    Returns:
        The relative L2 error and the solution's ``dtype``.
    """
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem

    domain = mesh.create_unit_square(comm, MESH_SIZE, MESH_SIZE)
    space = fem.functionspace(domain, ("Lagrange", DEGREE))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    coordinates = ufl.SpatialCoordinate(domain)
    exact = ufl.sin(math.pi * coordinates[0]) * ufl.sin(math.pi * coordinates[1])

    bilinear = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
    linear = ufl.inner(2 * math.pi**2 * exact, test) * ufl.dx

    condition = _dirichlet(domain, space, fem.Function(space))

    problem = LinearProblem(
        bilinear,
        linear,
        bcs=[condition],
        petsc_options_prefix="wheeltest_smoke_",
        petsc_options=dict(DIRECT_SOLVE),
    )
    solution = problem.solve()

    return _relative_error(domain, solution, exact), solution.x.array.dtype


def _dirichlet(domain: Any, space: Any, value: Any) -> Any:
    """Return the condition that imposes a function on the whole boundary."""
    from dolfinx import fem, mesh

    domain.topology.create_connectivity(1, 2)
    facets = mesh.exterior_facet_indices(domain.topology)
    return fem.dirichletbc(value, fem.locate_dofs_topological(space, 1, facets))


#: What each variant solves, and what its answer has to look like. The keys
#: are :data:`wheelbuild.petsc.SCALAR_TYPES`, so the flip that makes
#: ``dolfinx-solver-real`` moves the problem, the exact solution it is
#: measured against and the expected ``dtype`` together.
PROBLEMS = {
    "complex": Problem(
        name="complex Helmholtz",
        exact="the plane wave that solves it exactly",
        dtype_prefix="complex",
        solve=helmholtz,
    ),
    "real": Problem(
        name="real Poisson",
        exact="the sine product that solves it exactly",
        dtype_prefix="float",
        solve=poisson,
    ),
}


def problem_for(variant: str = petsc.SCALAR_TYPE) -> Problem:
    """Return the problem one scalar variant is proved by.

    Args:
        variant: The scalar type this distribution is built for, ``complex``
            or ``real``. It comes from the PETSc driver, so the flip that
            makes ``dolfinx-solver-real`` moves this stage with it.

    Returns:
        The problem.

    Raises:
        ValueError: When ``variant`` is not one of the two this build has a
            name for. Solving the complex problem under a real stack fails
            somewhere inside UFL instead, which says nothing.
    """
    try:
        return PROBLEMS[variant]
    except KeyError:
        raise ValueError(
            f"{variant!r} is not a scalar variant this build knows: "
            f"{', '.join(petsc.SCALAR_TYPES)} (spec §6)."
        ) from None


def proves(variant: str = petsc.SCALAR_TYPE) -> str:
    """Return what a passing smoke stage means, for the suite's own report.

    Args:
        variant: The scalar type this distribution is built for.

    Returns:
        One phrase, naming the variant's problem rather than a fixed one.
    """
    problem = problem_for(variant)
    return (
        f"a {problem.name} problem solves onto {problem.exact} and SLEPc "
        "finds the Dirichlet Laplacian's first eigenvalues"
    )


def error_problem(
    relative_error: float,
    *,
    variant: str = petsc.SCALAR_TYPE,
    tolerance: float = L2_TOLERANCE,
) -> str | None:
    """Report a solution that is not the exact one it should be.

    Args:
        relative_error: The discrete solution's L2 distance from the exact
            one, over the exact one's L2 norm.
        variant: The scalar type this distribution is built for.
        tolerance: How far it may be.

    Returns:
        A message, or ``None`` when the solve landed on the answer.
    """
    if relative_error <= tolerance:
        return None
    problem = problem_for(variant)
    return (
        f"the {problem.name} solve is {relative_error:.3e} away from "
        f"{problem.exact}, relative, against a tolerance of {tolerance:.0e}. "
        "The mesh and the element degree are fixed, so this is not "
        "discretisation error: the assembled operator, the linear solve or "
        f"the {variant} arithmetic under them is wrong. Note that the wheel "
        f"can import and report {variant} scalars and still fail here."
    )


def dtype_problem(dtype: Any, *, variant: str = petsc.SCALAR_TYPE) -> str | None:
    """Report a solution vector that does not hold the variant's scalars.

    Args:
        dtype: ``dtype`` of the solution's array.
        variant: The scalar type this distribution is built for.

    Returns:
        A message, or ``None`` when it is the variant's.
    """
    problem = problem_for(variant)
    if str(dtype).startswith(problem.dtype_prefix):
        return None
    return (
        f"the solution vector has dtype {dtype}, not a {problem.dtype_prefix} "
        f"one. This distribution's PETSc is built with {variant} scalars and "
        "DOLFINx bakes that type into every vector it hands back (spec §6); "
        "anything else here means the wheel was assembled from a prefix built "
        "for the other variant."
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
    condition = _dirichlet(domain, space, fem.Function(space))

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


def check(site: Path | None = None, *, variant: str = petsc.SCALAR_TYPE) -> str | None:
    """Solve both problems and report the first answer that is wrong.

    Args:
        site: Where the payload is installed. Used only to say where a
            failing import came from.
        variant: The scalar type this distribution is built for, which
            decides which problem is solved and what its answer has to be.

    Returns:
        A message, or ``None`` when the wheel computes what it should.
    """
    solve = problem_for(variant).solve
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

    relative_error, dtype = solve(comm)
    problem = dtype_problem(dtype, variant=variant)
    if problem is None:
        problem = error_problem(relative_error, variant=variant)
    if problem is None:
        problem = eigenvalue_problem(eigenvalues(comm))
    return problem


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the numerical smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument(
        "--scalar-type",
        default=petsc.SCALAR_TYPE,
        choices=list(petsc.SCALAR_TYPES),
        help="the variant the wheel under test was built for; defaults to "
        "the one the build drivers declare (spec §6)",
    )
    args = parser.parse_args(argv)

    problem = check(args.site, variant=args.scalar_type)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    solved = problem_for(args.scalar_type)
    print(
        f"{solved.name} solved to within {L2_TOLERANCE:.0e} of "
        f"{solved.exact}, and SLEPc found the Dirichlet Laplacian's first "
        f"{len(EXPECTED_EIGENVALUES)} eigenvalues"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
