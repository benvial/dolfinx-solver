"""Build KaHIP, the second graph partitioner DOLFINx can use.

PT-SCOTCH already satisfies DOLFINx's "at least one partitioner" requirement
and is what ParMETIS was excluded in favour of (spec §7). KaHIP is the optional
second one, and it is here because the licensing table says the wheel may ship
it (MIT) and spec §2 says not to trim a feature the licence allows: a user who
asks for ``dolfinx.mesh.create_mesh(..., graph.partitioner_kahip())`` on a
wheel built without it gets a runtime error, and PETSc's configure has no
``--download-kahip`` to fall back on.

Two of KaHIP's defaults have to be reversed for a wheel, and both are checked
rather than trusted:

* **``-march=native``.** KaHIP's CMake adds it unless ``NONATIVEOPTIMIZATIONS``
  is on. That is the OpenBLAS problem :mod:`wheelbuild.petsc` describes — a
  library tuned to the builder's CPU dies with an illegal instruction on an
  older one — so the flag is set and the built library is disassembled to
  confirm it took.
* **OpenMP.** KaHIP links it when CMake finds it, and ships an ``omp.h`` stub
  for when CMake does not. This stack has no threading anywhere in it by
  design (PETSc is configured ``--with-openmp=0``: a thread pool underneath an
  MPI rank oversubscribes the machine by the thread count, and the count comes
  from environment variables the wheel does not control), so the search is
  disabled and ``libgomp`` is asserted absent from the result.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from wheelbuild import elf, mpich
from wheelbuild._process import check_call

if TYPE_CHECKING:
    from collections.abc import Sequence

#: KaHIP release to build. Upstream DOLFINx's test environment pins this
#: release for the DOLFINx release this wheel mirrors
#: (``docker/Dockerfile.test-env``).
KAHIP_VERSION = "3.25"

#: What a finished prefix install has to contain. Both halves are required:
#: DOLFINx's ``FindKaHIP.cmake`` looks for ``parhip_interface.h`` and links
#: both ``libparhip_interface`` (the parallel partitioner) and ``libkahip``
#: (the serial one), and finding one without the other is a build whose MPI
#: half did not compile.
REQUIRED_ARTEFACTS = (
    Path("include/kaHIP_interface.h"),
    Path("include/parhip_interface.h"),
    Path("lib/libkahip.so"),
    Path("lib/libparhip_interface.so"),
)

#: The threading runtime that must not be linked in; see the module docstring.
OPENMP_SONAME = "libgomp.so.1"


class Build(NamedTuple):
    """What a built KaHIP prefix says about itself.

    Attributes:
        needed: ``DT_NEEDED`` entries of the installed parallel library, where
            an OpenMP runtime would show up and where the binding to MPI does.
        wide_vector_registers: Post-baseline vector register kinds the built
            code uses, from :func:`wheelbuild.elf.read_wide_vector_registers`.
            Empty unless a native-tuning flag survived.
    """

    needed: frozenset[str]
    wide_vector_registers: frozenset[str]


def source_url(version: str = KAHIP_VERSION) -> str:
    """Return the download URL of the pinned KaHIP source tarball."""
    return f"https://github.com/KaHIP/KaHIP/archive/v{version}.tar.gz"


def source_dir_name(version: str = KAHIP_VERSION) -> str:
    """Return the directory name the source tarball unpacks to.

    Args:
        version: The KaHIP release being built.

    Returns:
        The directory name, such as ``KaHIP-3.25`` — the repository's own
        capitalisation, which is not the tag's.
    """
    return f"KaHIP-{version}"


def configure_arguments(
    *, source_dir: Path, build_dir: Path, prefix: Path, mpi_prefix: Path
) -> list[str]:
    """Return the KaHIP CMake configure command.

    Args:
        source_dir: Unpacked KaHIP source tree.
        build_dir: Out-of-tree build directory.
        prefix: Shared install prefix.
        mpi_prefix: Prefix holding the MPI compiler wrappers.

    Returns:
        The ``cmake`` argument vector.
    """
    mpi_bin = mpi_prefix / "bin"
    return [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_C_COMPILER={mpi_bin / 'mpicc'}",
        f"-DCMAKE_CXX_COMPILER={mpi_bin / 'mpicxx'}",
        # The two reversals the module docstring explains.
        "-DNONATIVEOPTIMIZATIONS=ON",
        "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON",
        # ParHIP is the parallel partitioner DOLFINx actually calls, so the
        # MPI half is required rather than optional here.
        "-DNOMPI=OFF",
        "-DPARHIP=ON",
        # Named though they are already the defaults, per spec §8: 32-bit
        # indices match the PetscInt width the rest of the stack is built
        # with, Gurobi is a commercial solver the wheel cannot ship, and
        # neither tcmalloc nor KaHIP's own pybind11 module belongs in it.
        "-D64BITMODE=OFF",
        "-DDETERMINISTIC_PARHIP=OFF",
        "-DUSE_ILP=OFF",
        "-DUSE_TCMALLOC=OFF",
        "-DBUILDPYTHONMODULE=OFF",
    ]


def build_arguments(build_dir: Path, jobs: int) -> list[str]:
    """Return the command that compiles a configured KaHIP.

    Args:
        build_dir: The configured build directory.
        jobs: Parallel build jobs.

    Returns:
        The ``cmake --build`` argument vector.
    """
    return ["cmake", "--build", str(build_dir), "--parallel", str(jobs)]


def install_arguments(build_dir: Path) -> list[str]:
    """Return the command that installs a built KaHIP.

    Args:
        build_dir: The built build directory.

    Returns:
        The ``cmake --install`` argument vector.
    """
    return ["cmake", "--install", str(build_dir)]


def missing_artefacts(prefix: Path) -> list[Path]:
    """Return the required artefacts an install prefix does not have.

    Args:
        prefix: KaHIP install prefix.

    Returns:
        The missing paths, relative to the prefix, in declaration order.
    """
    return [
        relative for relative in REQUIRED_ARTEFACTS if not (prefix / relative).exists()
    ]


def observe(prefix: Path) -> Build:
    """Read what a built KaHIP prefix says about itself.

    Args:
        prefix: KaHIP install prefix.

    Returns:
        The facts :func:`build_problem` judges.

    Raises:
        subprocess.CalledProcessError: When the ELF tools cannot read the
            installed library.
    """
    library = prefix / "lib" / "libparhip_interface.so"
    _, needed = elf.read_dynamic(library)
    return Build(
        needed=needed,
        wide_vector_registers=frozenset(elf.read_wide_vector_registers(library)),
    )


def build_problem(build: Build, *, mpi_soname: str = mpich.MPI_SONAME) -> str | None:
    """Report a KaHIP that would not ship in a portable wheel.

    Args:
        build: What the install says about itself.
        mpi_soname: The MPI library the parallel partitioner has to ask for.

    Returns:
        A message naming the first thing that is wrong, or ``None``.
    """
    if build.wide_vector_registers:
        kinds = ", ".join(f"%{kind}" for kind in sorted(build.wide_vector_registers))
        return (
            f"libparhip_interface uses {kinds} registers, so it was compiled "
            "for a newer CPU than the x86-64 baseline — KaHIP's CMake adds "
            "-march=native unless NONATIVEOPTIMIZATIONS is on, and this "
            "build passes it. A wheel tuned to the builder's machine dies "
            "with an illegal instruction on an older one rather than running "
            "slowly (spec §2)."
        )

    if OPENMP_SONAME in build.needed:
        return (
            f"libparhip_interface asks the loader for {OPENMP_SONAME}. This "
            "stack has no threading in it: PETSc is configured "
            "--with-openmp=0 because a thread pool underneath an MPI rank "
            "oversubscribes the machine by a thread count the wheel does not "
            "control. KaHIP links OpenMP whenever CMake finds it, which is "
            "why the search is disabled rather than left alone."
        )

    if mpi_soname not in build.needed:
        found = ", ".join(sorted(build.needed)) or "nothing"
        return (
            f"libparhip_interface does not ask the loader for {mpi_soname} "
            f"(it needs {found}). ParHIP is the parallel half of KaHIP and "
            "the only half DOLFINx partitions a distributed mesh with; one "
            "that is not linked against MPI is a build whose MPI search "
            "silently failed."
        )
    return None


def validate(prefix: Path, build: Build, *, mpi_soname: str = mpich.MPI_SONAME) -> Path:
    """Check a built KaHIP against everything the wheel needs to be true.

    Args:
        prefix: KaHIP install prefix.
        build: What the install says about itself, from :func:`observe`.
        mpi_soname: The MPI library the parallel partitioner has to ask for.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is incomplete.
        ValueError: When the build would not ship in a portable wheel.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete KaHIP install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )

    problem = build_problem(build, mpi_soname=mpi_soname)
    if problem is not None:
        raise ValueError(problem)
    return prefix


def validate_install(prefix: Path, *, mpi_soname: str = mpich.MPI_SONAME) -> Path:
    """Read the facts about a built KaHIP, then :func:`validate` them.

    Args:
        prefix: KaHIP install prefix.
        mpi_soname: The MPI library the parallel partitioner has to ask for.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is incomplete.
        ValueError: When a check fails; see :func:`validate`.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete KaHIP install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )
    return validate(prefix, observe(prefix), mpi_soname=mpi_soname)


def run(
    *,
    source_dir: Path,
    build_dir: Path,
    prefix: Path,
    jobs: int,
    mpi_soname: str = mpich.MPI_SONAME,
) -> Path:
    """Configure, build, install and validate KaHIP.

    Args:
        source_dir: Unpacked KaHIP source tree.
        build_dir: Out-of-tree build directory.
        prefix: Shared install prefix, holding the MPI wrappers.
        jobs: Parallel build jobs.
        mpi_soname: The MPI library the parallel partitioner has to ask for.

    Returns:
        The validated install prefix.
    """
    check_call(
        configure_arguments(
            source_dir=source_dir,
            build_dir=build_dir,
            prefix=prefix,
            mpi_prefix=prefix,
        )
    )
    check_call(build_arguments(build_dir, jobs))
    check_call(install_arguments(build_dir))
    return validate_install(prefix, mpi_soname=mpi_soname)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the KaHIP build and its validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check an already-built prefix instead of building one",
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.validate_only and (args.source_dir is None or args.build_dir is None):
        parser.error("--source-dir and --build-dir are required to build")

    try:
        if args.validate_only:
            prefix = validate_install(args.prefix, mpi_soname=mpich.MPI_SONAME)
        else:
            prefix = run(
                source_dir=args.source_dir,
                build_dir=args.build_dir,
                prefix=args.prefix,
                jobs=args.jobs,
                mpi_soname=mpich.MPI_SONAME,
            )
    except (FileNotFoundError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"KaHIP {KAHIP_VERSION} installed into {prefix}: ParHIP linked "
        f"against {mpich.MPI_SONAME}, no OpenMP, no CPU tuning"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
