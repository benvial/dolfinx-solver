#!/usr/bin/env bash
# Run the wheel build in a manylinux_2_34 container on this machine.
#
#   scripts/build-in-container.sh
#
# The wheel targets manylinux_2_34_x86_64 (spec §2), so the vendored stack has
# to be compiled against that image's glibc rather than the host's. This is
# the only way to run the build locally, and it is the same entry point CI
# uses, so a failure here is the failure CI would see.
#
# Everything the build produces lives in ./.build-cache: the downloaded
# sources, the build trees, the shared install prefix and the ccache. A second
# run reuses all of it, which matters because a cold build of the dependency
# stack is a multi-hour compile.
#
# Environment:
#   MANYLINUX_IMAGE  image to build in (default quay.io/pypa/manylinux_2_34_x86_64)
#   BUILD_CACHE      host directory for the cached build state (default ./.build-cache)
#   JOBS             parallel build jobs (default: nproc)
#   DOCKER           container runtime (default docker; podman works too)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_34_x86_64}"
cache_dir="${BUILD_CACHE:-$repo_root/.build-cache}"
runtime="${DOCKER:-docker}"

mkdir -p "$cache_dir"

# The build runs as root inside the container because it installs the image's
# gfortran and ccache packages, so the cached tree ends up root-owned on the
# host. It is scratch state; delete it with sudo, or point BUILD_CACHE
# elsewhere.
exec "$runtime" run --rm \
  -v "$repo_root:/repo" \
  -v "$cache_dir:/build" \
  -e BUILD_ROOT=/build \
  -e JOBS="${JOBS:-$(nproc)}" \
  -w /repo \
  "$image" \
  /repo/scripts/build-wheel.sh "$@"
