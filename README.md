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

## Troubleshooting

**`import dolfinx` dies immediately, with UCX errors naming an `ib` or `mana`
device.** Something like:

```
ib_iface.c:1316 UCX  ERROR mana_0: ... failed: Operation not supported
```

MPICH chooses its transport through UCX when MPI initialises, which for this
wheel is the import itself. UCX looks at what the host exposes, and some hosts
expose an RDMA device a guest is not permitted to open — Azure's MANA card is
one, and a machine with it can kill the import before a single message has
been sent. It is not specific to running in parallel: a single-process script
that never calls MPI initialises it just the same.

Name the transports instead of letting UCX discover them:

```console
export UCX_TLS=self,sm,tcp
```

Shared memory, the intra-process transport and TCP are everything a
single-machine run needs, in serial or under `mpiexec -n N`. The wheel's own
test suite sets exactly this.

**On a cluster with working InfiniBand, do not set it.** There UCX finding the
card is the point, and naming TCP would be slower for no reason. That case is
untested: nothing in this project has ever run on a machine with a working
fabric, so the high-speed path is expected to work rather than known to.

## Releasing

A `v*` tag publishes three distributions from one workflow run: the two binary
variants it just built and tested, and the bare `dolfinx-solver`
meta-package, built in the publish job from `meta/`. They go up together —
`dolfinx-solver-real` cannot be added to a release afterwards under the same
version, and the meta-package pins its variant exactly — so the gather step
refuses a set that is not all three at `dolfinx_solver.__version__`:

```console
python -m wheelbuild.publish --wheelhouse wheelhouse --outdir dist
```

Each publish job runs that same gather over the whole release and then stages
only its own part with `--for`, because the upload publishes every file in the
directory it is given. A job that verified only its own distribution would
cheerfully upload half a release that can never be completed.

Uploading uses PyPI trusted publishing: an OIDC token minted for each publish
job, no API token stored anywhere.

**One publish job per distribution, each with its own environment.** That is
not a style choice. A *pending* publisher — the kind that registers a project
PyPI does not have yet — is identified by the claims in the OIDC token, which
carry the repository, the workflow and the environment but never the project
name. Three projects published from one repository, one workflow and one
environment are three identical configurations, and PyPI refuses the second
with *"a pending trusted publisher matching this configuration has already
been registered for a different project name"*. The environment is the field
left to tell them apart.

Register these three at <https://pypi.org/manage/account/publishing/>, owner
`benvial`, repository `dolfinx-solver`, workflow `wheels.yml`:

| PyPI project | Environment |
| --- | --- |
| `dolfinx-solver-complex` | `release-complex` |
| `dolfinx-solver-real` | `release-real` |
| `dolfinx-solver` | `release-meta` |

`wheelbuild.publish.ENVIRONMENTS` is where those names live, and the workflow
is asserted against it — so the file and the form cannot drift apart.

The matching GitHub environments already exist on the repository. They carry
no required reviewer; adding one to any of them is what makes that upload
pause for a human, which is worth considering given PyPI never reuses a
filename.

Nothing can be verified before the fact: PyPI has no API for reading
publishers back, and it does not check that the environments or the workflow
exist. The first tag is the test.

The tag is checked against the packaged version in seconds, in the checks job,
long before the build:

```console
git tag v$(python -c 'import dolfinx_solver; print(dolfinx_solver.__version__)')
```

After the upload, the `published` job installs the release the way a stranger
does — `pip install dolfinx-solver` on a machine that compiled nothing — and
puts three things to it: the meta-package resolves to
`dolfinx-solver-complex`, that wheel solves the numerical smoke problem, and
the refusals name the distribution pip recorded rather than the name that was
typed. It can also be run by hand against an index:

```console
python -m wheeltest.published --index pypi --work-dir /tmp/published
```

## License

LGPL-3.0-or-later, matching DOLFINx. The wheel carries a `THIRD-PARTY-NOTICES`
file with the license text of every vendored component.
