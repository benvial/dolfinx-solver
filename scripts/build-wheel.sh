#!/usr/bin/env bash
# Build the dolfinx-solver-complex wheel's vendored stack.
#
# Runs *inside* a manylinux_2_34 container; scripts/build-in-container.sh is
# what starts one. Each stage is a wheelbuild driver module (spec §8) invoked
# here with the paths this script owns, and each is skipped when its output is
# already in the cached install prefix — so a warm cache re-proves the stack
# instead of recompiling it.
#
# Stages so far: the Fortran half of MPICH, then PETSc with its whole
# --download-* dependency stack, then SLEPc against that PETSc. ADIOS2, KaHIP,
# petsc4py/slepc4py, DOLFINx and the wheel assembly follow in later tickets.
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

# Download and unpack a source tarball into $build_root, once. The marker is
# written only after tar returns, so an interrupted extraction is redone
# rather than compiled against half a source tree.
fetch_source() {
  local url="$1" source_dir="$2" archive="$build_root/${2##*/}.tar.gz"
  if [[ ! -f "$source_dir/.extracted" ]]; then
    rm -rf "${source_dir:?}"
    curl -fsSL "$url" -o "$archive"
    tar -xzf "$archive" -C "$build_root"
    touch "$source_dir/.extracted"
  fi
}

# Read a constant out of a wheelbuild driver, so the shell never holds a
# second copy of a version or a URL.
driver() { python -c "$1"; }

echo "==> toolchain"
# gfortran is the whole point of the MPICH stage, and the manylinux image does
# not carry it. flex is not in the image either and PT-SCOTCH's build refuses
# to configure without it, which PETSc reports hours in. ccache is what makes
# a warm rebuild cheap. cmake and bison the image already has, and PETSc drives
# its CMake sub-builds with -DCMAKE_POLICY_VERSION_MINIMUM=3.5, so the image's
# CMake 4 is happy with recipes written for CMake 2.8.
dnf install -y ccache gcc-gfortran patchelf flex >/dev/null
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
mpich_version="$(driver 'from wheelbuild.mpich import MPICH_VERSION; print(MPICH_VERSION)')"
mpich_url="$(driver 'from wheelbuild.mpich import source_url; print(source_url())')"
[[ -n "$mpich_version" ]] || { echo "could not read MPICH_VERSION" >&2; exit 1; }
mpich_source="$build_root/mpich-$mpich_version"
fetch_source "$mpich_url" "$mpich_source"
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

echo "==> PETSc (complex scalars, plus its whole --download-* dependency stack)"
# The multi-hour stage: OpenBLAS, ScaLAPACK, METIS, PT-SCOTCH, MUMPS,
# SuperLU_DIST and parallel HDF5 are all built by PETSc's configure from the
# one line in wheelbuild/petsc.py. PETSc builds in its own source tree, so
# $petsc_source is the build tree the cache is really keeping.
petsc_version="$(driver 'from wheelbuild.petsc import PETSC_VERSION; print(PETSC_VERSION)')"
petsc_url="$(driver 'from wheelbuild.petsc import source_url; print(source_url())')"
[[ -n "$petsc_version" ]] || { echo "could not read PETSC_VERSION" >&2; exit 1; }
petsc_source="$build_root/petsc-$petsc_version"
fetch_source "$petsc_url" "$petsc_source"
# The stamp, not a library file, is what says the stage is done: it is written
# after the driver has installed *and* validated, so an interrupted make
# install leaves no stamp and is rebuilt rather than re-checked forever. Its
# name carries the release and the scalar type, so bumping either takes the
# build path against a warm prefix instead of silently keeping the old one.
petsc_stamp="$install_prefix/.petsc-$petsc_version-complex.installed"
if [[ -f "$petsc_stamp" ]]; then
  echo "    cached in $install_prefix; re-checking what it says about itself"
  python -m wheelbuild.petsc --validate-only --prefix "$install_prefix"
else
  # The MPI prefix is the shared prefix: the wrappers, and the mpif.h PETSc's
  # Fortran packages compile against, are what the MPICH stage just installed.
  python -m wheelbuild.petsc \
    --source-dir "$petsc_source" \
    --prefix "$install_prefix" \
    --mpi-prefix "$install_prefix" \
    --jobs "$jobs"
  touch "$petsc_stamp"
fi

echo "==> SLEPc (against the PETSc just built)"
slepc_version="$(driver 'from wheelbuild.slepc import SLEPC_VERSION; print(SLEPC_VERSION)')"
slepc_url="$(driver 'from wheelbuild.slepc import source_url; print(source_url())')"
[[ -n "$slepc_version" ]] || { echo "could not read SLEPC_VERSION" >&2; exit 1; }
slepc_source="$build_root/slepc-$slepc_version"
fetch_source "$slepc_url" "$slepc_source"
# Stamped on the PETSc release too: a PETSc bump has to rebuild the SLEPc that
# was linked against the old one, and the stamp is what makes that automatic.
slepc_stamp="$install_prefix/.slepc-$slepc_version-petsc-$petsc_version.installed"
if [[ -f "$slepc_stamp" ]]; then
  echo "    cached in $install_prefix; re-checking it still binds that PETSc"
  python -m wheelbuild.slepc --validate-only --prefix "$install_prefix"
else
  python -m wheelbuild.slepc \
    --source-dir "$slepc_source" \
    --prefix "$install_prefix" \
    --jobs "$jobs"
  touch "$slepc_stamp"
fi

echo "==> done"
