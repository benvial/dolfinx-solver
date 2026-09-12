#!/usr/bin/env bash
# Shared front end for the wheel test scripts. Sourced, never run.
#
# Each script beside this one runs one stage of wheeltest.suite against a
# wheelhouse; all of them take the same arguments and differ only in which
# stage they name, so the argument handling lives here once.
#
# The wheel is tested from outside the container that built it: it is a
# manylinux_2_34 wheel, which is a promise that it runs on any glibc from
# 2.34 up, and running it only in the image it was compiled in would never
# test that promise. CI runs these scripts on the plain runner (spec §9).
#
# Environment:
#   WHEELHOUSE   directory holding the wheel (default ./.build-cache/wheelhouse,
#                which is where scripts/build-in-container.sh leaves it)
#   PYTHON       interpreter the clean venv is made from (default python3);
#                the matrix axis, since one abi3 wheel serves 3.12, 3.13, 3.14.
#                It needs nothing installed in it — the venv it seeds gets
#                everything from the wheel
#   DRIVER_PYTHON  interpreter the suite itself runs under (default python3).
#                Separate from PYTHON because the suite imports this
#                repository's driver packages and so needs the dev extras
#                (`pip install -e '.[dev]'`), while the matrix interpreter is
#                whatever bare python the wheel has to install into
#   WORK_DIR     scratch root for the venv and the stages (default
#                ./.build-cache/wheeltest)
#   DEMO_SOURCE  a DOLFINx source tree to take the demos from; downloaded
#                into $WORK_DIR/demo-source when unset
#   OFFLINE      set to 1 to resolve dependencies from $WHEELHOUSE rather
#                than from an index, which then has to hold mpich, mpi4py,
#                numpy, cffi, packaging and the upstream trio
#
# Anything passed on a script's command line is handed to wheeltest.suite.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wheelhouse="${WHEELHOUSE:-$repo_root/.build-cache/wheelhouse}"
work_dir="${WORK_DIR:-$repo_root/.build-cache/wheeltest}"
base_python="${PYTHON:-python3}"
driver_python="${DRIVER_PYTHON:-python3}"

# Sourcing keeps the sourcing script's positional parameters, so this is
# whatever the caller typed after the script name.
extra_arguments=("$@")

# The driver packages are imported from the repository and installed into
# nothing, exactly as the build does it.
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

# Run one stage of the suite against the wheelhouse.
#
#   run_stage <stage-name> [extra arguments for wheeltest.suite]
#
# The venv is made fresh per invocation: a reused one can still hold a
# previous wheel's files, which is how a suite comes to pass against
# something that is no longer there. Running several stages is therefore one
# call with several --stage arguments, which scripts/test-wheel.sh does.
run_stage() {
  local stages=() argument
  for argument in "$@"; do
    stages+=(--stage "$argument")
  done

  local options=()
  [[ -n "${DEMO_SOURCE:-}" ]] && options+=(--demo-source "$DEMO_SOURCE")
  [[ "${OFFLINE:-0}" == "1" ]] && options+=(--offline)

  "$driver_python" -m wheeltest.suite \
    --wheelhouse "$wheelhouse" \
    --venv "$work_dir/venv" \
    --work-dir "$work_dir" \
    --base-python "$base_python" \
    "${stages[@]}" \
    "${options[@]}" \
    "${extra_arguments[@]}"
}
