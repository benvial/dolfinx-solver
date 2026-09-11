#!/usr/bin/env bash
# Build the dolfinx-solver-complex wheel's vendored stack.
#
# Runs *inside* a manylinux_2_34 container; scripts/build-in-container.sh is
# what starts one. Each stage is a wheelbuild driver module (spec §8) invoked
# here with the paths this script owns, and each is skipped when its output is
# already in the cached install prefix — so a warm cache re-proves the stack
# instead of recompiling it.
#
# Stages so far: the Fortran half of MPICH. PETSc, SLEPc, ADIOS2, KaHIP,
# DOLFINx and the wheel assembly follow in later tickets.
#
# Environment:
#   BUILD_ROOT   scratch root for sources, build trees and the install prefix
#                (default /build); keep it on a cached volume
#   CCACHE_DIR   ccache directory (default $BUILD_ROOT/ccache)
#   JOBS         parallel build jobs (default: nproc)
#   PYTHON       interpreter the build venv is made from (default: the image's
#                cp312, which is the interpreter the abi3 wheel targets)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${BUILD_ROOT:-/build}"
jobs="${JOBS:-$(nproc)}"
base_python="${PYTHON:-/opt/python/cp312-cp312/bin/python}"
install_prefix="$build_root/install"
venv="$build_root/venv"

export CCACHE_DIR="${CCACHE_DIR:-$build_root/ccache}"
# The driver modules are imported from the repository, never installed: they
# are build tooling and ship in no wheel.
export PYTHONPATH="$repo_root"

mkdir -p "$build_root" "$CCACHE_DIR"

echo "==> toolchain"
# gfortran is the whole point of the MPICH stage, and the manylinux image does
# not carry it. ccache is what makes a warm rebuild cheap.
dnf install -y ccache gcc-gfortran patchelf >/dev/null
export PATH="/usr/lib64/ccache:$PATH"

echo "==> build environment"
[[ -x "$venv/bin/python" ]] || "$base_python" -m venv "$venv"
"$venv/bin/pip" install --quiet --upgrade pip
"$venv/bin/pip" install --quiet build auditwheel packaging wheel
# The PyPI mpich wheel is installed here for one reason: its libmpi.so.12 is
# the library a user's install resolves, and the MPICH stage proves the
# vendored libmpifort against that exact file (ADR-0001). It is installed
# under the bound the wheel declares, not unpinned: proving the Fortran half
# against a libmpi our own metadata would not allow proves nothing.
mpich_requirement="mpich$("$venv/bin/python" -c 'from wheelbuild.mpich import MPICH_VERSION
from wheelbuild.pin_check import expected_specifier
print(expected_specifier(MPICH_VERSION))')"
"$venv/bin/pip" install --quiet "$mpich_requirement"
export PATH="$venv/bin:$PATH"
runtime_libmpi="$venv/lib/libmpi.so.12"
# Later stages link against the vendored tree as it is being built.
export LD_LIBRARY_PATH="$install_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "==> pin check"
# Seconds, and it fails the run before the compile if the declared mpich bound
# and the MPICH the next stage builds have drifted apart.
python -m wheelbuild.pin_check

echo "==> install prefix layout"
# lib and lib64 become one directory before anything installs into either.
python -m wheelbuild.prefix --prefix "$install_prefix"

echo "==> MPICH (Fortran half only; the PyPI wheel supplies libmpi)"
mpich_version="$(python -c 'from wheelbuild.mpich import MPICH_VERSION; print(MPICH_VERSION)')"
mpich_url="$(python -c 'from wheelbuild.mpich import source_url; print(source_url())')"
[[ -n "$mpich_version" ]] || { echo "could not read MPICH_VERSION" >&2; exit 1; }
mpich_source="$build_root/mpich-$mpich_version"
# The marker is written only after tar returns, so an interrupted extraction
# is redone rather than compiled against half a source tree.
if [[ ! -f "$mpich_source/.extracted" ]]; then
  rm -rf "${mpich_source:?}"
  curl -fsSL "$mpich_url" -o "$build_root/mpich-$mpich_version.tar.gz"
  tar -xzf "$build_root/mpich-$mpich_version.tar.gz" -C "$build_root"
  touch "$mpich_source/.extracted"
fi
if [[ -f "$install_prefix/lib/libmpifort.so.12" ]]; then
  echo "    cached in $install_prefix; re-checking it against $runtime_libmpi"
  python -m wheelbuild.mpich \
    --validate-only \
    --prefix "$install_prefix" \
    --runtime-libmpi "$runtime_libmpi"
else
  # The build tree is scoped to the release, so a version bump configures a
  # fresh one instead of reconfiguring over the previous source tree.
  python -m wheelbuild.mpich \
    --source-dir "$mpich_source" \
    --build-dir "$build_root/mpich-$mpich_version-build" \
    --prefix "$install_prefix" \
    --runtime-libmpi "$runtime_libmpi" \
    --jobs "$jobs"
fi

echo "==> done"
