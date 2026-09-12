#!/usr/bin/env bash
# Run the whole wheel test suite against one clean venv.
#
#   scripts/test-wheel.sh
#
# Every stage the four scripts beside this one run, in one invocation and
# against one install — which is how CI runs it, since the venv and the
# install cost more than all the stages together. The stages run
# cheapest-first, so a wheel that is broken outright says so in seconds.
#
# See scripts/_wheel_venv.sh for the environment variables.
set -euo pipefail

# shellcheck source=scripts/_wheel_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_wheel_venv.sh"

run_stage imports foreign smoke interop demos
