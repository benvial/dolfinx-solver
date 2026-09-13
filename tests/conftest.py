"""Run the unit suite for the variant the repository declares.

`wheelbuild.petsc` resolves the scalar type from the environment, so that one
workflow can build `dolfinx-solver-complex` and `dolfinx-solver-real` from the
same drivers (ticket 21). The unit tests are not one of those builds: they
assert what the checked-in repository says — the name in `pyproject.toml`, the
meta-package's pin, the defines a correct PETSc reports — and those are facts
about the default variant. Left in place, an exported variable from a local
real build would turn a couple of dozen of them red for having read it.

So the variable is removed before any test module imports the drivers, and the
flip is tested where it is the subject rather than the weather:
`tests/test_real_variant.py` sets it deliberately and re-imports.
"""

import importlib
import os

from wheelbuild import petsc

# Reloaded rather than merely unset: the module resolves the variable once, at
# import, and importing it to learn the variable's name is already that
# import. Every module that reads the value reads it from here, and all of
# them are imported by the test modules below this file.
os.environ.pop(petsc.SCALAR_TYPE_VARIABLE, None)
importlib.reload(petsc)
