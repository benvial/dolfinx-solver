#!/usr/bin/env bash
# Build the dolfinx-solver-complex wheel's vendored stack.
#
# Runs *inside* a manylinux_2_34 container; scripts/build-in-container.sh is
# what starts one. Each stage is a wheelbuild driver module (spec §8) invoked
# here with the paths this script owns, and each is skipped when its output is
# already in the cached install prefix — so a warm cache re-proves the stack
# instead of recompiling it.
#
# Stages: the Fortran half of MPICH, then PETSc with its whole --download-*
# dependency stack, then SLEPc against that PETSc, then our own petsc4py and
# slepc4py against both, then ADIOS2 and KaHIP, then DOLFINx — its C++ core
# and its nanobind bindings — against all of it, and finally the wheel: the
# payload, the harvested notices, auditwheel repair, abi3audit, and an install
# into a clean venv that imports the stack back out.
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
# Boost (headers), pugixml and spdlog are DOLFINx's three required C++
# dependencies that nothing else in this build produces, and the licensing
# table already clears all three to ship (spec §7). pugixml and spdlog live in
# EPEL rather than the base image, which is why the repository is enabled
# first — the same sequence upstream's own wheel workflow uses.
dnf install -y dnf-plugins-core >/dev/null
dnf install -y epel-release >/dev/null
/usr/bin/crb enable
dnf install -y boost-devel pugixml-devel spdlog-devel >/dev/null
export PATH="/usr/lib64/ccache:$PATH"

echo "==> build environment"
[[ -x "$venv/bin/python" ]] || "$base_python" -m venv "$venv"
"$venv/bin/pip" install --quiet --upgrade pip
# build, auditwheel, abi3audit, packaging and wheel drive the packaging;
# setuptools, Cython and numpy are what petsc4py's and slepc4py's setup.py
# need, and they are installed here rather than fetched per build so
# --no-build-isolation can keep the bindings' build on versions this build
# controls. mpi4py is the runtime half of the import check: it is what dlopens
# the PyPI mpich wheel's libmpi before any compiled module of ours (spec §5).
"$venv/bin/pip" install --quiet \
  build auditwheel abi3audit packaging wheel setuptools "cython>=3" \
  "numpy>=2" mpi4py "scikit-build-core>=0.11"
# nanobind is pinned, not floored: our DOLFINx bindings and the published
# fenics-basix extension share a nanobind type registry only when their ABI
# tags agree, and a mismatch surfaces as a TypeError in the user's first
# functionspace call. wheelbuild.dolfinx holds the pin and the check.
"$venv/bin/pip" install --quiet "$("$venv/bin/python" -c 'from wheelbuild.dolfinx import NANOBIND_REQUIREMENT
print(NANOBIND_REQUIREMENT)')"
# The upstream trio is not vendored and never re-pinned here: DOLFINx's C++
# core compiles against Basix's headers and FFCx's ufcx.h, and the releases it
# gets are the ones this wheel declares a runtime dependency on, read straight
# out of our own pyproject.toml (spec §10).
# shellcheck disable=SC2046 # each requirement is one word by construction
"$venv/bin/pip" install --quiet $("$venv/bin/python" -c 'from wheelbuild.dolfinx import upstream_trio_requirements
print(" ".join(upstream_trio_requirements()))')
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

# The scalar type is read once, here, and every stage below takes it from
# this variable: the stamp names, the PETSc driver's flag, and the claim the
# prefix is about to make. The real variant (spec §6) is the flip of
# wheelbuild.petsc.SCALAR_TYPE and nothing else in this script.
scalar_type="$(driver 'from wheelbuild.petsc import SCALAR_TYPE; print(SCALAR_TYPE)')"
[[ -n "$scalar_type" ]] || { echo "could not read SCALAR_TYPE" >&2; exit 1; }

echo "==> install prefix layout"
# lib and lib64 become one directory before anything installs into either, and
# the prefix records which variant owns it. A prefix holds one scalar type:
# PetscScalar is baked into libpetsc, the bindings and DOLFINx alike, and the
# stamps below are never pruned — so building the other variant in here would
# find this one's stamps and re-check a stack of the wrong scalar type. The
# driver refuses that and says to point BUILD_ROOT at a second directory, one
# per variant. A warm prefix from before the marker existed is adopted by the
# first run that claims it, which is the complex build this repository has
# been running all along.
python -m wheelbuild.prefix --prefix "$install_prefix" \
  --scalar-type "$scalar_type"

echo "==> MPICH (binding shims only; the PyPI wheel supplies libmpi)"
mpich_version="$(driver 'from wheelbuild.mpich import MPICH_VERSION; print(MPICH_VERSION)')"
mpich_url="$(driver 'from wheelbuild.mpich import source_url; print(source_url())')"
[[ -n "$mpich_version" ]] || { echo "could not read MPICH_VERSION" >&2; exit 1; }
mpich_source="$build_root/mpich-$mpich_version"
fetch_source "$mpich_url" "$mpich_source"
# A stamp, not one of the installed shims, is what says the stage is done —
# the same marker the stages below use. An installed library appears partway
# through make install, so an install interrupted between libmpifort and
# libmpicxx left a prefix that took the cached path on every later run and
# then failed --validate-only for the missing half, forever. The stamp is
# written after the driver has installed *and* validated, and it carries
# MPICH_VERSION, so a release bump against a warm prefix rebuilds rather than
# keeping shims built against another libmpi's symbols.
mpich_stamp="$install_prefix/.mpich-$mpich_version.installed"
if [[ -f "$mpich_stamp" ]]; then
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
  touch "$mpich_stamp"
fi

echo "==> PETSc ($scalar_type scalars, plus its whole --download-* dependency stack)"
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
# Both come from the driver: a scalar type spelled out here could not change
# when wheelbuild.petsc.SCALAR_TYPE does, and the real variant (spec §6) is
# exactly that flip — which against a warm prefix would have found the complex
# stamp, re-checked it, and failed for the wrong reason.
#
# Deliberately not stamped on the MPICH release, though this stage is compiled
# with the prefix's wrappers (ticket 19). ADIOS2's and KaHIP's stamps do name
# it, and the difference is cost, not correctness: their rebuilds are minutes.
# The correctness argument is that a bump inside the pinned series moves
# nothing this stage's output depends on — every library here asks the loader
# for libmpi.so.12 and, in a user's install, is answered by the PyPI mpich
# wheel's copy rather than by anything built here (spec §5, ADR-0001), so being
# linked against one 5.x libmpi rather than another is not a property the wheel
# preserves. Moving the pin out of the series is what the pin check refuses.
petsc_stamp="$install_prefix/.petsc-$petsc_version-$scalar_type.installed"
if [[ -f "$petsc_stamp" ]]; then
  echo "    cached in $install_prefix; re-checking what it says about itself"
  python -m wheelbuild.petsc --validate-only --prefix "$install_prefix" \
    --scalar-type "$scalar_type"
else
  # The MPI prefix is the shared prefix: the wrappers, and the mpif.h PETSc's
  # Fortran packages compile against, are what the MPICH stage just installed.
  python -m wheelbuild.petsc \
    --source-dir "$petsc_source" \
    --scalar-type "$scalar_type" \
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
# On its scalar type as well, since that is baked into the binary and a flip
# of it leaves the release untouched. Not on the MPICH release, for the reason
# spelled out at the PETSc stamp above: the loader answers libmpi.so.12 with
# the PyPI mpich wheel's copy, so an in-series bump relinks nothing here
# (spec §5, ADR-0001).
slepc_stamp="$install_prefix/.slepc-$slepc_version-petsc-$petsc_version-$scalar_type.installed"
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

echo "==> petsc4py + slepc4py (ours, against the vendored PETSc and SLEPc)"
# Both bindings come out of the tarballs the PETSc and SLEPc stages already
# downloaded, never the PyPI sdists, and are staged into $install_prefix/python
# under their upstream import names with no dist-info — the layout the wheel
# assembly step grafts and the layout their relative rpath is written for.
# Stamped on both releases and on the scalar type, since a bump or a flip of
# any of them rebuilds both bindings. Not on the MPICH release: the extensions
# reach MPI through libpetsc and through mpi4py's own libmpi, and the copy that
# answers libmpi.so.12 at run time is the PyPI mpich wheel's rather than the
# one in this prefix (spec §5, ADR-0001), so an in-series bump is not an input
# to what this stage produces. Same reading as the PETSc stamp above.
bindings_stamp="$install_prefix/.bindings-petsc-$petsc_version-$scalar_type-slepc-$slepc_version.installed"
if [[ -f "$bindings_stamp" ]]; then
  echo "    cached in $install_prefix; re-checking the staged tree and importing it"
  python -m wheelbuild.bindings --validate-only --prefix "$install_prefix"
else
  python -m wheelbuild.bindings \
    --prefix "$install_prefix" \
    --petsc-source "$petsc_source" \
    --slepc-source "$slepc_source"
  touch "$bindings_stamp"
fi

echo "==> ADIOS2 (parallel I/O, against the prefix's parallel HDF5)"
adios2_version="$(driver 'from wheelbuild.adios2 import ADIOS2_VERSION; print(ADIOS2_VERSION)')"
adios2_url="$(driver 'from wheelbuild.adios2 import source_url; print(source_url())')"
adios2_dir="$(driver 'from wheelbuild.adios2 import source_dir_name; print(source_dir_name())')"
[[ -n "$adios2_version" ]] || { echo "could not read ADIOS2_VERSION" >&2; exit 1; }
adios2_source="$build_root/$adios2_dir"
fetch_source "$adios2_url" "$adios2_source"
# Stamped on two releases it does not build: the PETSc whose configure put the
# parallel HDF5 in this prefix, which wheelbuild/adios2.py configures against
# with HDF5_ROOT and HDF5_PREFER_PARALLEL, and the MPICH whose mpicc/mpicxx
# compile it — libadios2_cxx_mpi is linked against that MPICH's headers and that
# libmpi's exported symbols, and comes out carrying libmpicxx.so.12 in its
# DT_NEEDED (spec §5, ADR-0001: the soname is frozen across the series, the
# exported symbols are not). A bump of either release moves something ADIOS2 was
# linked against, and adios2.validate reads the installed features and sonames
# rather than whether what they were linked against is still there, so without
# these names a warm prefix would re-check a stale ADIOS2 instead of relinking
# it. Not stamped on the scalar type: this is the one PETSc-bound stage that
# never links libpetsc, and the HDF5 it does link is the same library either way.
adios2_stamp="$install_prefix/.adios2-$adios2_version-petsc-$petsc_version-mpich-$mpich_version.installed"
if [[ -f "$adios2_stamp" ]]; then
  echo "    cached in $install_prefix; re-checking what it says about itself"
  python -m wheelbuild.adios2 --validate-only --prefix "$install_prefix"
else
  python -m wheelbuild.adios2 \
    --source-dir "$adios2_source" \
    --build-dir "$build_root/$adios2_dir-build" \
    --prefix "$install_prefix" \
    --jobs "$jobs"
  touch "$adios2_stamp"
fi

echo "==> KaHIP (the optional second partitioner beside PT-SCOTCH)"
kahip_version="$(driver 'from wheelbuild.kahip import KAHIP_VERSION; print(KAHIP_VERSION)')"
kahip_url="$(driver 'from wheelbuild.kahip import source_url; print(source_url())')"
kahip_dir="$(driver 'from wheelbuild.kahip import source_dir_name; print(source_dir_name())')"
[[ -n "$kahip_version" ]] || { echo "could not read KAHIP_VERSION" >&2; exit 1; }
kahip_source="$build_root/$kahip_dir"
fetch_source "$kahip_url" "$kahip_source"
# Stamped on the MPICH release as well: ParHIP is compiled with this prefix's
# mpicc/mpicxx, against that MPICH's headers and that libmpi's exported
# symbols, which is the one input from the prefix KaHIP has. The soname in
# libparhip_interface's DT_NEEDED is not the argument — it stays libmpi.so.12
# across the series (ADR-0001) — the symbols behind it are. Nothing else here
# comes from the prefix: KaHIP builds no PETSc package and links no libpetsc,
# so neither the PETSc release nor its scalar type belongs in this name.
kahip_stamp="$install_prefix/.kahip-$kahip_version-mpich-$mpich_version.installed"
if [[ -f "$kahip_stamp" ]]; then
  echo "    cached in $install_prefix; re-checking it is portable and MPI-linked"
  python -m wheelbuild.kahip --validate-only --prefix "$install_prefix"
else
  python -m wheelbuild.kahip \
    --source-dir "$kahip_source" \
    --build-dir "$build_root/$kahip_dir-build" \
    --prefix "$install_prefix" \
    --jobs "$jobs"
  touch "$kahip_stamp"
fi

echo "==> DOLFINx (C++ core, then the cp312-abi3 nanobind bindings)"
# The product. The C++ core installs into the shared prefix beside everything
# it links; the bindings are built by scikit-build-core with
# wheel.py-api=cp312 and staged into $install_prefix/python beside petsc4py
# and slepc4py, which is the layout the wheel assembly grafts.
dolfinx_version="$(driver 'from wheelbuild.dolfinx import DOLFINX_VERSION; print(DOLFINX_VERSION)')"
dolfinx_url="$(driver 'from wheelbuild.dolfinx import source_url; print(source_url())')"
dolfinx_dir="$(driver 'from wheelbuild.dolfinx import source_dir_name; print(source_dir_name())')"
[[ -n "$dolfinx_version" ]] || { echo "could not read DOLFINX_VERSION" >&2; exit 1; }
dolfinx_source="$build_root/$dolfinx_dir"
fetch_source "$dolfinx_url" "$dolfinx_source"
# Stamped on every release DOLFINx is linked against, not only its own, and on
# the scalar type of the PETSc among them:
# PetscScalar is baked into libdolfinx and into the bindings, and KaHIP's two
# libraries have unversioned sonames, so a bump there would relink silently.
# The nanobind pin is in the name too, since that is what decides whether the
# bindings can exchange types with the basix wheel. A bump of any of them has
# to rebuild, not re-check.
# The bindings' build tree is keyed on the pin too: CMake caches the nanobind
# it was configured against, so a bump has to configure a fresh tree rather
# than relink against the old one.
# Not stamped on the MPICH release, for the reason at the PETSc stamp above:
# libdolfinx names libmpi.so.12 in its DT_NEEDED and a user's install resolves
# that to the PyPI mpich wheel, not to the libmpi this prefix holds (spec §5,
# ADR-0001), so an in-series bump leaves what this stage produces unchanged.
nanobind_version="$(driver 'from wheelbuild.dolfinx import NANOBIND_VERSION; print(NANOBIND_VERSION)')"
dolfinx_stamp="$install_prefix/.dolfinx-$dolfinx_version-petsc-$petsc_version-$scalar_type-slepc-$slepc_version-adios2-$adios2_version-kahip-$kahip_version-nanobind-$nanobind_version.installed"
if [[ -f "$dolfinx_stamp" ]]; then
  echo "    cached in $install_prefix; re-checking the build and importing it"
  python -m wheelbuild.dolfinx --validate-only --prefix "$install_prefix"
else
  python -m wheelbuild.dolfinx \
    --source-dir "$dolfinx_source" \
    --cpp-build-dir "$build_root/$dolfinx_dir-cpp-build" \
    --python-build-dir "$build_root/$dolfinx_dir-python-build-nanobind-$nanobind_version" \
    --prefix "$install_prefix" \
    --jobs "$jobs"
  touch "$dolfinx_stamp"
fi

echo "==> wheel (payload, notices, repair, audits, clean-venv install)"
# The step that turns the prefix into the artefact. It is minutes rather than
# hours, and every part of it is a check on the stages above — the payload is
# the DT_NEEDED closure of the three extensions, the notices are harvested
# from the source trees those stages downloaded, and the finished wheel is
# installed into a fresh venv and imported. So it runs on every build, warm
# cache or not, rather than being skipped by a stamp.
#
# The wheelhouse is not $install_prefix/wheelhouse: that one holds the
# petsc4py, slepc4py and fenics_dolfinx wheels the earlier stages built, and
# those are build artefacts that must never be published (spec §6, §10).
wheelhouse="$build_root/wheelhouse"
python -m wheelbuild.assemble \
  --prefix "$install_prefix" \
  --build-root "$build_root" \
  --wheelhouse "$wheelhouse" \
  --base-python "$base_python"

echo "==> done"
