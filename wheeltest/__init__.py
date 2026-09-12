"""Prove a built wheel works, from outside the build that produced it.

:mod:`wheelbuild` checks a wheel the way its author can: against the install
prefix it came from, inside the container that compiled it, with the source
trees still on disk. This package checks it the way a user meets it — a file
in a wheelhouse, installed into an empty venv on a machine that has none of
that — which is the only vantage point from which some failures are visible
at all. The wheel that passed every build-side audit could not
``import dolfinx.fem.petsc`` (ticket 16), because nothing inside the build
had reason to.

The suite is five stages, run by :mod:`wheeltest.suite` against one venv:

* **imports** — :mod:`wheelbuild.import_check`, pointed at the installed
  site rather than the staging site. Same code, and deliberately so: one
  MPI runtime in the process, complex scalars, DOLFINx reporting the feature
  set it was compiled with.
* **foreign** — :mod:`wheeltest.foreign`, which builds the environments
  ``dolfinx_solver._bootstrap`` exists to refuse and checks that it refuses
  them, and that it lets a clean one through.
* **smoke** — :mod:`wheeltest.smoke`, a complex Helmholtz solve against a
  known answer and a SLEPc eigensolve against a known eigenvalue.
* **interop** — :mod:`wheeltest.interop` under ``mpiexec -n 2``, where
  mpi4py, petsc4py and DOLFINx have to agree on one communicator.
* **demos** — :mod:`wheeltest.demos`, a subset of upstream's own demo
  programs run against the wheel, serial and parallel.

Every stage reports its problems as written sentences rather than
tracebacks, for the reason the build drivers do: the three ways this can
fail — the install, the import, the arithmetic — have different causes and
different people fix them, so a run has to say which one happened.
"""
