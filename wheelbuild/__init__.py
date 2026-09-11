"""Build-time tooling for the dolfinx-solver wheels.

Nothing here ships inside a wheel: these modules build the vendored stack
(MPICH's Fortran bindings, PETSc, SLEPc, ADIOS2, DOLFINx), harvest third-party
notices, assemble and repair the platform wheel, and guard the release.
"""
