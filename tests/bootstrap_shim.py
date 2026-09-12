"""Load `dolfinx_solver._bootstrap` without importing the package around it.

`dolfinx_solver/__init__.py` calls `bootstrap()` in its module body — that is
the whole point of the shim, ordering `mpi4py.MPI` ahead of every compiled
module (spec §5, ADR-0002) — so `from dolfinx_solver import _bootstrap`
initialises a real MPI runtime as a side effect of collecting a test file.

These tests do not want one. They drive the guards with fakes, on a machine
where no wheel is installed and no payload exists; the import that matters is
proved against an installed wheel by `wheeltest`, which is the only vantage
point from which it means anything. On a GitHub runner the side effect is not
merely unnecessary: MPI initialisation killed the whole pytest process with
SIGTERM and no output at all, which is a long afternoon to read backwards.

So the module is loaded from its own file, under a name of its own, with no
parent package executed.
"""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "dolfinx_solver" / "_bootstrap.py"
_SPEC = importlib.util.spec_from_file_location("dolfinx_solver_bootstrap", _PATH)
assert _SPEC is not None
assert _SPEC.loader is not None

bootstrap_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bootstrap_module)
