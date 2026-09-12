#!/usr/bin/env bash
# Solve two problems with known answers against the installed wheel.
#
#   scripts/smoke-test.sh
#
# A complex Helmholtz problem whose exact solution is a plane wave, and the
# Dirichlet Laplacian's first eigenvalues through SLEPc. Between them they
# put the vendored MUMPS, the complex scalar type and the PETSc layer of
# the payload on the path of a single run (spec §11).
#
# See scripts/_wheel_venv.sh for the environment variables.
set -euo pipefail

# shellcheck source=scripts/_wheel_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_wheel_venv.sh"

run_stage smoke
