#!/usr/bin/env bash
# Run the installed wheel under mpiexec -n 2.
#
#   scripts/interop-test.sh
#
# mpi4py, petsc4py and DOLFINx in one process, sharing one communicator
# over a mesh the vendored PT-SCOTCH partitioned, with the launcher coming
# from the same PyPI mpich wheel that supplies the libmpi they all bind to
# (spec §5, §11; ADR-0002). Nothing serial can show any of that.
#
# See scripts/_wheel_venv.sh for the environment variables.
set -euo pipefail

# shellcheck source=scripts/_wheel_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_wheel_venv.sh"

run_stage interop
