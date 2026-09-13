# dolfinx-solver

[DOLFINx](https://github.com/FEniCS/dolfinx) (FEniCSx) packaged as a Linux
binary wheel, with a **complex-scalar PETSc and SLEPc inside the wheel** — the
piece upstream's own experimental wheels leave out.

```console
pip install dolfinx-solver          # the complex build, via the meta-package
pip install dolfinx-solver-complex  # the same wheel, named directly
```

```python
import dolfinx_solver  # noqa: F401 - orders MPI ahead of the bindings
import dolfinx
from petsc4py import PETSc

assert PETSc.ScalarType.__name__ == "complex128"
```

Import `dolfinx_solver` **before** `dolfinx`, `petsc4py` or `slepc4py`.
That import loads `mpi4py.MPI` first, so every later binding binds to the
`libmpi.so.12` mpi4py loaded, and it fails with an explanation if a foreign
petsc4py would shadow the vendored one.

## What is in the wheel

DOLFINx's C++ core and nanobind bindings, complex-scalar PETSc and SLEPc with
petsc4py and slepc4py built against exactly those, and the vendored native
stack (HDF5-parallel, SuperLU_DIST, MUMPS, PT-SCOTCH, OpenBLAS, ADIOS2 and
friends). `fenics-basix`, `fenics-ffcx` and `fenics-ufl` come from PyPI at
upstream's own pins — they are never re-vendored.

MPI is **not** vendored: the wheel depends on the PyPI
[`mpich`](https://pypi.org/project/mpich/) wheel and carries only MPICH's
Fortran and C++ binding libraries, which that wheel does not ship. DOLFINx,
mpi4py and petsc4py therefore all share one MPI in one process.

## Requirements and constraints

- Linux x86_64, `manylinux_2_34`, Python 3.12+ (one `cp312-abi3` wheel).
- The PyPI `mpich` wheel, 5.0 series (installed for you). The wheel vendors
  MPICH's binding libraries built against that series, so neither an older
  `mpich` nor another MPI — Open MPI, Intel MPI — will do.
- A virtual-environment-style prefix; install into an environment of its own.
- Do not co-install the PyPI `petsc`, `petsc4py`, `slepc` or `slepc4py`
  projects — they would put a second PETSc in the process.
- Install one scalar variant, never both: the two wheels are the same payload
  at the same paths, so together they overwrite each other's files. The import
  refuses that environment rather than running whichever one landed second.

The real-scalar build is published separately as `dolfinx-solver-real`; PETSc's
scalar type is baked into the binaries, so it cannot be switched at runtime.

## Versioning

The version mirrors the DOLFINx release the wheel ships, and the `.postN`
segment moves forward for packaging-only fixes:
`dolfinx_solver.__version__` is the single source, `DOLFINX_VERSION` names the
upstream tag it builds.

## Development

```console
pip install -e '.[dev]'
ruff check . && ruff format --check .
mypy dolfinx_solver wheelbuild wheeltest
pytest
```

`wheelbuild/` holds the build drivers and ships in no wheel; `scripts/` drives
the manylinux container build.

### Testing a built wheel

`wheeltest/` is the suite that proves a finished wheel from outside the build:
it installs it into a clean venv and puts five stages to it. Point it at a
wheelhouse — `scripts/build-in-container.sh` leaves one in
`.build-cache-<variant>` —
and run the whole thing, or one stage at a time:

```console
scripts/test-wheel.sh                 # every stage, one venv
scripts/wheel-test.sh                 # install, import order, foreign petsc4py
scripts/smoke-test.sh                 # the variant's problem and an eigensolve
scripts/interop-test.sh               # mpiexec -n 2, one shared communicator
scripts/demo-test.sh                  # five upstream DOLFINx demos
```

`PYTHON` picks the interpreter the clean venv is made from (the wheel is abi3
and has to work on 3.12, 3.13 and 3.14, and that interpreter needs nothing
installed in it), `WHEELHOUSE` the directory holding the wheel, and
`DEMO_SOURCE` a DOLFINx source tree to take the demos from instead of
downloading the pinned release. The suite itself runs under `DRIVER_PYTHON`,
which is the environment the dev extras are installed in.

## License

LGPL-3.0-or-later, matching DOLFINx. The wheel carries a `THIRD-PARTY-NOTICES`
file with the license text of every vendored component.
