# dolfinx-solver

[![Build](https://img.shields.io/github/actions/workflow/status/benvial/dolfinx-solver/wheels.yml?style=for-the-badge&logo=github&label=build)](https://github.com/benvial/dolfinx-solver/actions/workflows/wheels.yml)
[![PYPI](https://img.shields.io/pypi/v/dolfinx-solver?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/dolfinx-solver)
[![LICENSE](https://img.shields.io/pypi/l/dolfinx-solver?style=for-the-badge&logo=readthedocs&logoColor=white&color=blue)](https://github.com/benvial/dolfinx-solver/blob/main/LICENSE)

[DOLFINx](https://github.com/FEniCS/dolfinx) (FEniCSx) packaged as a Linux
binary wheel, with a **complex-scalar PETSc and SLEPc inside the wheel**.

```console
pip install dolfinx-solver          # the complex build, via the meta-package
pip install dolfinx-solver-complex  # the same wheel, named directly
```

```python
import dolfinx_solver  # noqa: F401 - import first; it orders MPI
import dolfinx
from petsc4py import PETSc

assert PETSc.ScalarType.__name__ == "complex128"
```

Import `dolfinx_solver` **before** `dolfinx`, `petsc4py` or `slepc4py`. It
loads `mpi4py.MPI` first, so every later binding binds to the same
`libmpi.so.12`, and it fails with an explanation if a foreign petsc4py would
shadow the vendored one.

## Scalar variants

One release publishes three distributions:

| Distribution | Installs | `PETSc.ScalarType` |
| --- | --- | --- |
| `dolfinx-solver` | the meta-package, which resolves to the complex build | — |
| `dolfinx-solver-complex` | complex scalars | `complex128` |
| `dolfinx-solver-real` | real scalars | `float64` |

The real-scalar build is published separately as `dolfinx-solver-real`; PETSc's
scalar type is baked into the binaries, so it cannot be switched at runtime.
Install one of the two, never both: they are the same payload at the same
paths, so together they overwrite each other's files, and the import refuses
that environment rather than running whichever landed second.

Each distribution's project page is this page, retargeted to the build it
describes — which is why the snippet above asserts one scalar type rather than
listing both.

## Installing

- Linux x86_64, `manylinux_2_34`, Python 3.12+ — one `cp312-abi3` wheel.
- Install into an environment of its own.
- MPI comes from the PyPI [`mpich`](https://pypi.org/project/mpich/) wheel,
  5.0 series, installed for you. Nothing else works: the wheel vendors MPICH's
  Fortran and C++ binding libraries built against that series, so neither an
  older `mpich` nor another MPI will do.
- Do not co-install the PyPI `petsc`, `petsc4py`, `slepc` or `slepc4py`
  projects; they would put a second PETSc in the process.

## What is in the wheel

DOLFINx's C++ core and nanobind bindings, complex-scalar PETSc and SLEPc with
petsc4py and slepc4py built against exactly those, and the vendored native
stack (HDF5-parallel, SuperLU_DIST, MUMPS, PT-SCOTCH, OpenBLAS, ADIOS2 and
friends). `fenics-basix`, `fenics-ffcx` and `fenics-ufl` come from PyPI at
upstream's own pins.

MPI is **not** vendored, so DOLFINx, mpi4py and petsc4py share one MPI in one
process.

The version mirrors the DOLFINx release the wheel ships, and the `.postN`
segment moves forward for packaging-only fixes:
`dolfinx_solver.__version__` is the single source, `DOLFINX_VERSION` names the
upstream tag it builds.

## Troubleshooting

**The import dies with UCX errors naming an `ib` or `mana` device.** MPICH
chooses its transport through UCX when MPI initialises, which for this wheel is
the import itself, and some hosts expose an RDMA device a guest may not open —
Azure's MANA card is one. It is not specific to running in parallel. Name the
transports instead of letting UCX discover them:

```console
export UCX_TLS=self,sm,tcp
```

Shared memory, the intra-process transport and TCP cover any single-machine
run, serial or under `mpiexec -n N`; the test suite sets exactly this. **On a
cluster with working InfiniBand, do not set it** — there, UCX finding the card
is the point. That path is untested: no machine in this project has a working
fabric.

## Development

```console
pip install -e '.[dev]'
ruff check . && ruff format --check .
mypy dolfinx_solver wheelbuild wheeltest
pytest
```

`wheelbuild/` holds the build drivers and ships in no wheel; `scripts/` drives
the manylinux container build. `scripts/build-in-container.sh` builds a wheel
on this machine and leaves a wheelhouse in `.build-cache-<variant>`.

### Testing a built wheel

`wheeltest/` tests a finished wheel from outside the build: it installs it into
a clean venv and runs five stages against it. The whole suite, or one stage:

| Command | What it runs |
| --- | --- |
| `scripts/test-wheel.sh` | every stage, one venv |
| `scripts/wheel-test.sh` | install, import order, foreign petsc4py |
| `scripts/smoke-test.sh` | the variant's problem and an eigensolve |
| `scripts/interop-test.sh` | `mpiexec -n 2`, one shared communicator |
| `scripts/demo-test.sh` | five upstream DOLFINx demos |

`WHEELHOUSE` is the directory holding the wheel. `PYTHON` picks the interpreter
the clean venv is made from — the wheel is abi3 and has to work on 3.12, 3.13
and 3.14, and that interpreter needs nothing installed in it. `DRIVER_PYTHON`
is the environment the dev extras are in, and `DEMO_SOURCE` a DOLFINx source
tree to take the demos from instead of downloading the pinned release.

## Releasing

A `v*` tag publishes all three distributions from one workflow run: the two
binary variants it just built and tested, and the meta-package, built from
`meta/`. They go up together, and the meta-package pins its variant exactly, so
the gather step refuses a set that is not all three at
`dolfinx_solver.__version__`:

```console
python -m wheelbuild.publish --wheelhouse wheelhouse --outdir dist
```

Each publish job gathers the whole release and stages only its own part with
`--for`, because the upload publishes every file in the directory it is given.

Uploading uses PyPI trusted publishing, **one job and one GitHub environment
per distribution**. A pending publisher is identified by the claims in the OIDC
token, which carry the repository, the workflow and the environment but not the
project name; three projects published from one repository and workflow would
be three identical configurations, and PyPI refuses the second. Register these
at <https://pypi.org/manage/account/publishing/>, owner `benvial`, repository
`dolfinx-solver`, workflow `wheels.yml`:

| PyPI project | Environment |
| --- | --- |
| `dolfinx-solver-complex` | `release-complex` |
| `dolfinx-solver-real` | `release-real` |
| `dolfinx-solver` | `release-meta` |

`wheelbuild.publish.ENVIRONMENTS` holds those names and the workflow is
asserted against it, so the file and the form cannot drift apart. The GitHub
environments carry no required reviewer; adding one makes that upload wait for
a human, which matters because PyPI never reuses a filename. None of it can be
verified up front — PyPI has no API for reading publishers back — so the first
tag is the test.

The tag is checked against the packaged version in the checks job, before the
build:

```console
git tag v$(python -c 'import dolfinx_solver; print(dolfinx_solver.__version__)')
```

Noticing that upstream has released is a job of its own:
`.github/workflows/dolfinx-release.yml` asks GitHub weekly for DOLFINx's latest
release and, when it is ahead, writes that release into the four files carrying
it — the version module, the source digest, upstream's trio pins and the
meta-package — then opens the result as a pull request for `wheels.yml` to
build. It needs a `RELEASE_BUMP_TOKEN` secret, because a pull request opened
with the default token triggers no workflows. The nanobind coupling, the MPICH
series and the tag itself are left to a human.

After the upload, the `published` job installs the release the way a stranger
does — `pip install dolfinx-solver` on a machine that compiled nothing — and
checks three things: the meta-package resolves to `dolfinx-solver-complex`,
that wheel solves the numerical smoke problem, and the refusals name the
distribution pip recorded rather than the name that was typed. By hand, against
any index:

```console
python -m wheeltest.published --index pypi --work-dir /tmp/published
```

## License

LGPL-3.0-or-later, matching DOLFINx. The wheel carries a `THIRD-PARTY-NOTICES`
file with the license text of every vendored component.
