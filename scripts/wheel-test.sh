#!/usr/bin/env bash
# Wheel-level checks: install the wheel into a clean venv, import the stack
# out of it, and put the environments the wheel refuses to the installed
# copy.
#
#   scripts/wheel-test.sh
#
# This is the stage that proves the packaging rather than the mathematics:
# the metadata brings the PyPI mpich wheel and the upstream trio with the
# install, mpi4py reaches the loader before any compiled module, the
# vendored libraries resolve through the rpaths they carry, PETSc reports
# complex scalars, and dolfinx_solver refuses an environment holding a
# second PETSc while letting a clean one through (spec §11).
#
# See scripts/_wheel_venv.sh for the environment variables.
set -euo pipefail

# shellcheck source=scripts/_wheel_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_wheel_venv.sh"

run_stage imports foreign
