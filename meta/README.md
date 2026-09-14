# dolfinx-solver

`pip install dolfinx-solver` installs [`dolfinx-solver-complex`](https://pypi.org/project/dolfinx-solver-complex/),
the complex-scalar build of DOLFINx (FEniCSx) with PETSc and SLEPc inside the
wheel. This project ships no binaries of its own — it exists so the bare name
resolves to a working default.

PETSc's scalar type is baked into the binaries, so the real-scalar build is a
separate distribution, [`dolfinx-solver-real`](https://pypi.org/project/dolfinx-solver-real/),
released alongside this one. Install that name directly when you want real
scalars.

Linux x86_64 only. See the [repository](https://github.com/benvial/dolfinx-solver)
for the installation constraints (MPICH-family MPI, one environment per
install).
