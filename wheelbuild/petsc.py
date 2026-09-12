"""Build the complex-scalar PETSc the rest of the wheel is compiled against.

PETSc is not one dependency here, it is the dependency stack. One configure
line downloads and builds OpenBLAS, ScaLAPACK, METIS, PT-SCOTCH, MUMPS,
SuperLU_DIST and parallel HDF5, using recipes PETSc's own developers keep
working against the PETSc release they ship with (spec §8). Building those
seven separately would mean owning seven sets of build flags and the
compatibility between them; letting PETSc drive them means owning one line,
recorded verbatim in :func:`configure_arguments`.

The line is maximal and explicit. Every option that matters to the wheel is
named even where it repeats a PETSc default, because a default is a decision
someone else can change between releases, and this build has to produce the
same ABI every time it runs. Scalar type is the one axis left as a parameter:
``PetscScalar`` is baked into ``libpetsc``, into petsc4py's extension and into
DOLFINx's binaries alike, so the real variant is this same driver with
``--with-scalar-type=real`` and a different distribution name (spec §6), never
a runtime switch.

Two exclusions are policy rather than taste. **ParMETIS is never built**: its
licence forbids redistribution without permission from the University of
Minnesota, so a wheel carrying it could not be published at all, and parallel
partitioning and MUMPS's parallel ordering come from PT-SCOTCH instead (spec
§7). **OpenBLAS is built with dynamic architecture dispatch**, because
PETSc's recipe otherwise lets OpenBLAS tune itself to whatever CPU the build
ran on — fine for a cluster install compiled in place, fatal for a wheel that
is downloaded onto machines older than the builder's, where the result is an
illegal instruction rather than a slow solve.

What the build produced is then read back out of the installed
``petscconf.h`` and the installed ``libpetsc``, for the reason ticket 02 left
on the record: a fact read from a file cannot be shadowed by another copy of
the same software elsewhere on the machine, and a fact obtained by running
something out of the prefix can be.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from wheelbuild import elf, macros
from wheelbuild._process import check_call

if TYPE_CHECKING:
    from collections.abc import Sequence

#: PETSc release to build. The whole stack — SLEPc, petsc4py, slepc4py, the
#: DOLFINx build — is pinned to this one minor series per release (spec §6),
#: because petsc4py pins PETSc at minor granularity and SLEPc's minor tracks
#: PETSc's. Moving this means moving :mod:`wheelbuild.slepc` with it.
PETSC_VERSION = "3.25.5"

#: The scalar type this distribution is built for. ``dolfinx-solver-complex``
#: is the first release; ``dolfinx-solver-real`` is this flipped to ``real``.
SCALAR_TYPE = "complex"

#: The two scalar variants this build has a name for. Both distributions are
#: built from these drivers, and a spelling outside this pair is a typo that
#: would otherwise be read as "not complex" by every check that compares
#: against one of them (spec §6).
SCALAR_TYPES = ("complex", "real")

#: Real precision. Single-precision PETSc is a different ABI again and no
#: variant of this wheel ships it.
PRECISION = "double"

#: Optimisation level for every language. ``-O2`` with no ``-march``: the
#: wheel is manylinux_2_34_x86_64 and has to run on every x86-64 machine, so
#: nothing may be tuned to the builder's CPU. Upstream DOLFINx's own PETSc
#: images use the same level.
OPTIMISATION_FLAGS = "-O2"

#: How OpenBLAS is told to build for an unknown machine. ``DYNAMIC_ARCH``
#: compiles every x86-64 kernel and chooses at load time, ``DYNAMIC_OLDER``
#: adds the pre-AVX kernels to that set, and ``TARGET`` fixes the baseline
#: build so no part of it is tuned by CPU detection on the build machine.
#: Without this PETSc's recipe builds whatever the builder's CPU supports and
#: the wheel dies with an illegal instruction on an older one.
OPENBLAS_MAKE_OPTIONS = "DYNAMIC_ARCH=1 DYNAMIC_OLDER=1 TARGET=PRESCOTT"

#: The stack PETSc's configure downloads and builds for us (spec §8). Every
#: one of these is a licence the wheel may ship (spec §7); ParMETIS, which is
#: not, is absent by design and asserted absent by :func:`build_problem`.
DOWNLOADED_PACKAGES = (
    "openblas",
    "scalapack",
    "metis",
    "ptscotch",
    "mumps",
    "superlu_dist",
    "hdf5",
)

#: What a finished prefix install has to contain. ``petscvariables`` is how
#: PETSc's own check target and DOLFINx's build find the configuration, so a
#: prefix without it is a build that never reached ``make install``.
REQUIRED_ARTEFACTS = (
    Path("include/petsc.h"),
    Path("include/petscconf.h"),
    Path("include/petscversion.h"),
    Path("lib/libpetsc.so"),
    Path("lib/petsc/conf/petscvariables"),
)

#: Features the wheel is built to have, and the ``petscconf.h`` define that
#: proves each one. "Never trim a solver feature for packaging convenience"
#: (spec §2): a PETSc that silently configured without MUMPS because a Fortran
#: compiler went missing is a wheel that has quietly lost its direct solver.
REQUIRED_DEFINES = {
    "PETSC_HAVE_OPENBLAS": "OpenBLAS",
    "PETSC_HAVE_SCALAPACK": "ScaLAPACK",
    "PETSC_HAVE_METIS": "METIS",
    "PETSC_HAVE_PTSCOTCH": "PT-SCOTCH",
    "PETSC_HAVE_MUMPS": "MUMPS",
    "PETSC_HAVE_SUPERLU_DIST": "SuperLU_DIST",
    "PETSC_HAVE_HDF5": "HDF5",
}

#: Defines that must not appear, and why each one would sink the wheel.
FORBIDDEN_DEFINES = {
    "PETSC_HAVE_PARMETIS": (
        "ParMETIS may not be redistributed without permission from the "
        "University of Minnesota, so a wheel carrying it cannot be published "
        "at all. Parallel partitioning and MUMPS's parallel ordering come "
        "from PT-SCOTCH instead (spec §7)"
    ),
    "PETSC_HAVE_MPIUNI": (
        "MPIUNI is PETSc's sequential stub MPI, which configure falls back to "
        "when it cannot find a real one. A wheel built against it would run "
        "every solve on one rank while mpi4py reported many"
    ),
    "PETSC_USE_DEBUG": (
        "a debugging PETSc checks every argument of every call and is several "
        "times slower; it is not what a release wheel ships"
    ),
    "PETSC_USE_64BIT_INDICES": (
        "PetscInt width is part of the ABI that libpetsc, petsc4py's "
        "extension and the DOLFINx binaries all share, and this stack is "
        "built 32-bit throughout"
    ),
}

#: The define that records the scalar type, present only for complex builds
#: (PETSc's ``config/PETSc/options/scalarTypes.py``). The real variant is the
#: same driver asserting its absence.
COMPLEX_DEFINE = "PETSC_USE_COMPLEX"

#: The define that records double precision.
PRECISION_DEFINE = "PETSC_USE_REAL_DOUBLE"

#: Where a real ParMETIS would show up in the prefix. Deliberately narrow:
#: PT-SCOTCH installs ``libptscotchparmetis*``, which is its own emulation of
#: the ParMETIS *interface* under the CeCILL-C licence and not ParMETIS code,
#: so a pattern matching every path containing "parmetis" would fail the build
#: over exactly the library we chose ParMETIS's replacement for.
PARMETIS_PATTERNS = ("lib/libparmetis.*", "include/parmetis.h")


class Build(NamedTuple):
    """What a built PETSc prefix says about itself.

    Attributes:
        version: The release, from ``include/petscversion.h``.
        defines: The names ``petscconf.h`` defines, which is where configure
            records every feature and ABI choice of the build.
        soname: ``DT_SONAME`` of the installed ``libpetsc``.
        needed: ``DT_NEEDED`` entries of the installed ``libpetsc``.
    """

    version: str | None
    defines: frozenset[str]
    soname: str | None
    needed: frozenset[str]


def series(version: str) -> str:
    """Return the PETSc minor series a release belongs to.

    Args:
        version: A PETSc release, such as ``3.25.5``.

    Returns:
        The ``major.minor`` series, such as ``3.25`` — the granularity
        everything downstream couples at.
    """
    major, _, rest = version.partition(".")
    return f"{major}.{rest.partition('.')[0]}"


def source_url(version: str = PETSC_VERSION) -> str:
    """Return the download URL of the pinned PETSc source tarball."""
    return (
        "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/"
        f"petsc-{version}.tar.gz"
    )


def soname(version: str = PETSC_VERSION) -> str:
    """Return the soname the built ``libpetsc`` declares.

    PETSc versions its shared library by the minor series, not the release, so
    a patch bump keeps the soname and everything already linked against it.

    Args:
        version: The PETSc release being built.

    Returns:
        The soname, such as ``libpetsc.so.3.25``.
    """
    return f"libpetsc.so.{series(version)}"


def build_arch(scalar_type: str = SCALAR_TYPE) -> str:
    """Return the ``PETSC_ARCH`` a build of one scalar type uses.

    A prefix install still builds through an arch directory, and naming it
    after the scalar type keeps the two variants' build trees apart in a cache
    that may hold both.

    Args:
        scalar_type: ``complex`` or ``real``.

    Returns:
        The arch name, such as ``arch-complex``.
    """
    return f"arch-{scalar_type}"


def configure_arguments(
    *,
    source_dir: Path,
    prefix: Path,
    mpi_prefix: Path,
    jobs: int,
    scalar_type: str = SCALAR_TYPE,
) -> list[str]:
    """Return the PETSc ``configure`` command, in full.

    This is the recorded line the ticket asks for: nothing about the build is
    decided anywhere else, and nothing relevant is left to a PETSc default.

    Args:
        source_dir: Unpacked PETSc source tree.
        prefix: Shared install prefix — the same one MPICH installed into.
        mpi_prefix: Prefix holding the MPI compiler wrappers, which is that
            same shared prefix.
        jobs: Parallel build jobs for PETSc's own sub-builds.
        scalar_type: ``complex`` for this distribution, ``real`` for the
            variant that ships later.

    Returns:
        The ``configure`` argument vector.
    """
    mpi_bin = mpi_prefix / "bin"
    return [
        str(source_dir / "configure"),
        f"PETSC_ARCH={build_arch(scalar_type)}",
        f"--prefix={prefix}",
        # The MPI is the Fortran-enabled MPICH ticket 02 put in this prefix:
        # its mpif.h is what MUMPS, ScaLAPACK and SuperLU_DIST compile
        # against. The wrappers are named one by one rather than passing
        # --with-mpi-dir, because that prefix is also where PETSc is about to
        # install itself and pointing PETSc's MPI search at its own install
        # directory invites it to find the wrong things there.
        "--with-mpi=1",
        f"--with-cc={mpi_bin / 'mpicc'}",
        f"--with-cxx={mpi_bin / 'mpicxx'}",
        f"--with-fc={mpi_bin / 'mpifort'}",
        f"--with-mpiexec={mpi_bin / 'mpiexec'}",
        f"--with-make-np={jobs}",
        # The ABI axes. Scalar type is the parameter the real variant flips;
        # the other three are the same in every variant this project ships,
        # and are stated so that a change in PETSc's defaults cannot move them.
        f"--with-scalar-type={scalar_type}",
        f"--with-precision={PRECISION}",
        "--with-64-bit-indices=0",
        "--with-64-bit-blas-indices=0",
        # A release build of shared libraries: the wheel vendors .so files and
        # the assembly step repairs them, and a debugging PETSc would ship a
        # solver several times slower than the one users expect.
        "--with-debugging=0",
        "--with-shared-libraries=1",
        f"--COPTFLAGS={OPTIMISATION_FLAGS}",
        f"--CXXOPTFLAGS={OPTIMISATION_FLAGS}",
        f"--FOPTFLAGS={OPTIMISATION_FLAGS}",
        # Parallelism is MPI's, all of it. A threaded BLAS underneath an MPI
        # rank oversubscribes the machine by the thread count, and the number
        # of threads would then depend on environment variables the wheel does
        # not control, so neither OpenMP nor pthreads is built into the stack.
        "--with-openmp=0",
        "--download-openblas-use-pthreads=0",
        f"--download-openblas-make-options={OPENBLAS_MAKE_OPTIONS}",
        # Nothing in the wheel calls PETSc, HDF5 or SuperLU_DIST from Fortran
        # — DOLFINx is C++ and the bindings are Python — so their Fortran
        # layers are build time this stack does not have to spend. The Fortran
        # compiler is still needed: MUMPS, ScaLAPACK and OpenBLAS are written
        # in it.
        "--with-fortran-bindings=0",
        "--with-hdf5-fortran-bindings=0",
        "--with-superlu_dist-fortran-bindings=0",
        # No X11 in a container, and no graphics in a wheel.
        "--with-x=0",
        # ParMETIS is excluded outright; see FORBIDDEN_DEFINES.
        "--with-parmetis=0",
        *(f"--download-{package}=1" for package in DOWNLOADED_PACKAGES),
    ]


def missing_artefacts(prefix: Path) -> list[Path]:
    """Return the required artefacts an install prefix does not have.

    Args:
        prefix: PETSc install prefix.

    Returns:
        The missing paths, relative to the prefix, in declaration order.
    """
    return [
        relative for relative in REQUIRED_ARTEFACTS if not (prefix / relative).exists()
    ]


def parmetis_artefacts(prefix: Path) -> list[Path]:
    """Return any ParMETIS the prefix carries.

    Args:
        prefix: Install prefix to search.

    Returns:
        The paths found, relative to the prefix, sorted. PT-SCOTCH's
        ``libptscotchparmetis*`` interface emulation is not among them; see
        :data:`PARMETIS_PATTERNS`.
    """
    found = {
        path.relative_to(prefix)
        for pattern in PARMETIS_PATTERNS
        for path in prefix.glob(pattern)
    }
    return sorted(found)


def version_in_header(header_text: str) -> str | None:
    """Return the release an installed ``petscversion.h`` describes.

    Args:
        header_text: Contents of the installed ``include/petscversion.h``.

    Returns:
        The release, such as ``3.25.5``, or ``None`` when the header does not
        carry all three version components.
    """
    defines = macros.defined_macros(header_text)
    components = []
    for component in ("MAJOR", "MINOR", "SUBMINOR"):
        value = defines.get(f"PETSC_VERSION_{component}")
        if value is None:
            return None
        components.append(value)
    return ".".join(components)


def observe(prefix: Path) -> Build:
    """Read what a built PETSc prefix says about itself.

    Args:
        prefix: PETSc install prefix.

    Returns:
        The facts :func:`build_problem` judges.

    Raises:
        FileNotFoundError: When a header the facts come from is missing.
        subprocess.CalledProcessError: When the ELF tools cannot read
            ``libpetsc``.
    """
    library_soname, needed = elf.read_dynamic(prefix / "lib" / "libpetsc.so")
    return Build(
        version=version_in_header(
            (prefix / "include" / "petscversion.h").read_text(encoding="utf-8")
        ),
        defines=macros.read_defined_names(prefix / "include" / "petscconf.h"),
        soname=library_soname,
        needed=needed,
    )


def build_problem(
    build: Build,
    *,
    expected_version: str = PETSC_VERSION,
    scalar_type: str = SCALAR_TYPE,
) -> str | None:
    """Report a PETSc that is not the one this wheel is built around.

    Args:
        build: What the install says about itself.
        expected_version: The release this driver is pinned to.
        scalar_type: The scalar type this variant is built for.

    Returns:
        A message naming the first thing that is wrong, or ``None`` when the
        build matches on every count.
    """
    problems = (
        _release_problem(build, expected_version),
        _scalar_problem(build, scalar_type),
        _precision_problem(build),
        _feature_problem(build),
    )
    return next((problem for problem in problems if problem is not None), None)


def _release_problem(build: Build, expected_version: str) -> str | None:
    """Report a prefix holding a PETSc other than the pinned release."""
    if build.version is None:
        return (
            "the installed petscversion.h carries no PETSC_VERSION_MAJOR/"
            "MINOR/SUBMINOR, so nothing here can confirm which PETSc this "
            "prefix holds."
        )
    if build.version != expected_version:
        return (
            f"this prefix holds PETSc {build.version}, but the wheel is built "
            f"against PETSc {expected_version}. petsc4py pins PETSc at minor "
            "granularity and SLEPc's minor tracks PETSc's, so the whole stack "
            "moves together or not at all (spec §6)."
        )

    expected_soname = soname(expected_version)
    if build.soname != expected_soname:
        return (
            f"the installed libpetsc declares the soname {build.soname!r} "
            f"rather than {expected_soname}. petsc4py's extension, slepc4py's "
            "and the DOLFINx bindings all reach one libpetsc by that name at "
            "load time, and two copies under different names would be two "
            "PETSc global states in one process."
        )
    return None


def _precision_problem(build: Build) -> str | None:
    """Report a build whose real precision is not the one the wheel ships."""
    if PRECISION_DEFINE in build.defines:
        return None
    return (
        f"petscconf.h does not define {PRECISION_DEFINE}, so this is not a "
        f"{PRECISION}-precision build. Precision is part of the PetscScalar "
        "ABI every binary in the wheel shares."
    )


def _feature_problem(build: Build) -> str | None:
    """Report a solver the build is missing, or a define that sinks the wheel."""
    for define, component in REQUIRED_DEFINES.items():
        if define not in build.defines:
            return (
                f"this PETSc was configured without {component} "
                f"(petscconf.h does not define {define}). A package configure "
                "could not build is a line in a log and a solver missing from "
                f"the wheel, not a failed run — read configure.log for what "
                f"{component} tripped over. Do not publish without it "
                "(spec §2)."
            )

    for define, reason in FORBIDDEN_DEFINES.items():
        if define in build.defines:
            return f"petscconf.h defines {define}: {reason}."
    return None


def _scalar_problem(build: Build, scalar_type: str) -> str | None:
    """Report a build whose ``PetscScalar`` is not the variant's scalar type."""
    is_complex = COMPLEX_DEFINE in build.defines
    if is_complex == (scalar_type == "complex"):
        return None
    built = "complex" if is_complex else "real"
    presence = "present" if is_complex else "absent"
    return (
        f"this PETSc is a {built}-scalar build and this distribution is the "
        f"{scalar_type}-scalar one ({COMPLEX_DEFINE} is {presence} in "
        "petscconf.h). PetscScalar is baked into libpetsc, petsc4py's "
        "extension and DOLFINx's binaries alike, so the scalar type is the "
        "distribution, not a setting (spec §6)."
    )


def validate(prefix: Path, build: Build, *, scalar_type: str = SCALAR_TYPE) -> Path:
    """Check a built PETSc against everything the wheel needs to be true.

    Args:
        prefix: PETSc install prefix.
        build: What the install says about itself, from :func:`observe`.
        scalar_type: The scalar type this variant is built for.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is incomplete.
        ValueError: When the build is not the PETSc this wheel is built
            around, or when ParMETIS is in the tree.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete PETSc install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )

    parmetis = parmetis_artefacts(prefix)
    if parmetis:
        carried = ", ".join(str(relative) for relative in parmetis)
        raise ValueError(
            f"this prefix carries ParMETIS ({carried}). "
            f"{FORBIDDEN_DEFINES['PETSC_HAVE_PARMETIS']}."
        )

    problem = build_problem(build, scalar_type=scalar_type)
    if problem is not None:
        raise ValueError(problem)
    return prefix


def validate_install(prefix: Path, *, scalar_type: str = SCALAR_TYPE) -> Path:
    """Read the facts about a built PETSc, then :func:`validate` them.

    Args:
        prefix: PETSc install prefix.
        scalar_type: The scalar type this variant is built for. It has to
            travel with the prefix: the check that the install is the right
            scalar type is the one check whose answer depends on which
            distribution is being built, and defaulting it would pass a real
            build off as a complex one.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is incomplete.
        ValueError: When a check fails; see :func:`validate`.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete PETSc install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )
    return validate(prefix, observe(prefix), scalar_type=scalar_type)


def check_arguments(prefix: Path) -> list[str]:
    """Return PETSc's own check target, aimed at a prefix install.

    PETSc prints this line at the end of ``make install``. It compiles and
    runs a handful of examples against the installed libraries — including an
    MPI one — which is the cheapest end-to-end proof that the stack this
    driver just built actually links and runs.

    Args:
        prefix: PETSc install prefix.

    Returns:
        The ``make`` argument vector, to be run in the PETSc source tree.
    """
    return ["make", f"PETSC_DIR={prefix}", "PETSC_ARCH=", "check"]


def run(
    *,
    source_dir: Path,
    prefix: Path,
    mpi_prefix: Path,
    jobs: int,
    scalar_type: str = SCALAR_TYPE,
) -> Path:
    """Configure, build, install, check and validate PETSc.

    PETSc builds in its own source tree rather than out of tree, so
    ``source_dir`` is also the build directory and is what a warm cache keeps:
    the downloaded OpenBLAS, MUMPS and friends live under its arch directory
    and are what make a cold run take hours.

    Args:
        source_dir: Unpacked PETSc source tree.
        prefix: Shared install prefix.
        mpi_prefix: Prefix holding the MPI compiler wrappers.
        jobs: Parallel build jobs.
        scalar_type: The scalar type to build.

    Returns:
        The validated install prefix.
    """
    arch = build_arch(scalar_type)
    check_call(
        configure_arguments(
            source_dir=source_dir,
            prefix=prefix,
            mpi_prefix=mpi_prefix,
            jobs=jobs,
            scalar_type=scalar_type,
        ),
        cwd=source_dir,
    )
    make = ["make", f"PETSC_DIR={source_dir}", f"PETSC_ARCH={arch}"]
    check_call([*make, f"-j{jobs}", "all"], cwd=source_dir)
    check_call([*make, "install"], cwd=source_dir)
    check_call(check_arguments(prefix), cwd=source_dir)
    return validate_install(prefix, scalar_type=scalar_type)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the PETSc build and its validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument(
        "--mpi-prefix",
        type=Path,
        help="prefix holding the MPI wrappers; defaults to --prefix",
    )
    parser.add_argument("--scalar-type", default=SCALAR_TYPE, choices=SCALAR_TYPES)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check an already-built prefix instead of building one",
    )
    # Not needed by --validate-only, which is how a warm build cache re-proves
    # a prefix it did not have to rebuild.
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.validate_only and args.source_dir is None:
        parser.error("--source-dir is required to build")

    try:
        if args.validate_only:
            prefix = validate_install(args.prefix, scalar_type=args.scalar_type)
        else:
            prefix = run(
                source_dir=args.source_dir,
                prefix=args.prefix,
                mpi_prefix=args.mpi_prefix or args.prefix,
                jobs=args.jobs,
                scalar_type=args.scalar_type,
            )
    except (FileNotFoundError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"PETSc {PETSC_VERSION} ({args.scalar_type} scalars) installed into "
        f"{prefix} with {', '.join(REQUIRED_DEFINES.values())}, and no ParMETIS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
