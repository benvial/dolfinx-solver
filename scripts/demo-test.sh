#!/usr/bin/env bash
# Run a subset of upstream's own demos against the installed wheel.
#
#   scripts/demo-test.sh
#
# Five demos from the DOLFINx release this wheel mirrors, two of them under
# two ranks and two of them complex-scalar (spec §11). They are code
# written without any knowledge of this packaging, which is the point: they
# reach parts of the library a hand-written smoke test has no reason to.
#
# The sources are not in the wheel. Set DEMO_SOURCE to a DOLFINx source
# tree — the container build leaves one in .build-cache — or let the run
# download the pinned release once into $WORK_DIR/demo-source.
#
# See scripts/_wheel_venv.sh for the environment variables.
set -euo pipefail

# shellcheck source=scripts/_wheel_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_wheel_venv.sh"

run_stage demos
