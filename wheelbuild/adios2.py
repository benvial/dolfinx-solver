"""Build the ADIOS2 that backs DOLFINx's parallel I/O.

ADIOS2 is the one piece of DOLFINx's I/O stack that PETSc's configure does not
supply, and it is what `dolfinx.io.VTXWriter` and the checkpoint writers are
compiled against. DOLFINx asks for it as ``find_package(ADIOS2 2.8.1 COMPONENTS
CXX MPI)``, so the build has to produce the MPI-enabled C++ library, not the
serial one.

The configure line is maximal and explicit for a reason particular to this
project. Nearly every ADIOS2 feature defaults to ``AUTO``, which means "enable
it if the machine happens to have the library" — so the same source tree
produces a different ADIOS2, with a different set of vendored dependencies,
depending on what is installed in the build image. A wheel cannot be built that
way twice: every enabled feature is a shared library the assembly step grafts
and a licence the notice harvester has to account for (spec §7). So every
optional feature this wheel does not ship is named ``OFF`` in
:data:`DISABLED_FEATURES` rather than left to the search, and the same list is
then asserted absent in the installed ``ADIOSConfig.h``.

What is on: MPI and HDF5 — the parallel engines DOLFINx uses, against the same
parallel HDF5 PETSc's configure already put in the prefix. The BP3/BP4/BP5 file
engines ADIOS2 always compiles in are what DOLFINx's writers actually use; the
streaming transports (SST, DataMan, ZeroMQ) and the compressor plugins are
external-dependency features no DOLFINx API reaches, which is why turning them
off costs the wheel nothing.
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

#: ADIOS2 release to build. Upstream DOLFINx's own test environment pins this
#: release for the DOLFINx release this wheel mirrors
#: (``docker/Dockerfile.test-env``), which is the closest thing to a supported
#: pairing that exists.
ADIOS2_VERSION = "2.12.1"

#: The C++ library DOLFINx links against. ADIOS2 2.11 renamed it from
#: ``adios2_cxx11`` to ``adios2_cxx``, which is why DOLFINx's CMake looks for
#: both targets; the wheel builds one pinned release, so one name is what this
#: driver checks for.
CXX_LIBRARY = "libadios2_cxx_mpi"

#: What a finished prefix install has to contain. The ``_mpi`` libraries are
#: the ones DOLFINx links: ADIOS2 builds a serial and an MPI variant of each
#: library, and a prefix carrying only the serial ones is a build whose MPI
#: search quietly failed.
REQUIRED_ARTEFACTS = (
    Path("include/adios2.h"),
    Path("include/adios2/common/ADIOSConfig.h"),
    Path("lib/libadios2_core_mpi.so"),
    Path(f"lib/{CXX_LIBRARY}.so"),
    Path("lib/cmake/adios2/adios2-config.cmake"),
)

#: Features the wheel is built to have, spelled as ADIOS2's CMake spells them.
#: Both are passed ``ON`` rather than left at ``AUTO``, so a missing MPI or a
#: missing parallel HDF5 fails this stage instead of producing a quietly
#: reduced ADIOS2 that DOLFINx then fails to find.
ENABLED_FEATURES = ("MPI", "HDF5")

#: Features turned off by name. Every one of them is either an external
#: library this wheel does not ship (the compressors, the streaming
#: transports, Catalyst, the cloud back-ends) or a binding no part of this
#: wheel calls (Python, Fortran). Left at their ``AUTO`` default they would be
#: switched on by whatever the build image happens to carry, which is how a
#: wheel ends up with a vendored library nobody chose and no licence text.
DISABLED_FEATURES = (
    "AWSSDK",
    "BigWhoop",
    "Blosc2",
    "BZip2",
    "Caliper",
    "Campaign",
    "Catalyst",
    "CUDA",
    "CURL",
    "DAOS",
    "DataMan",
    "DataSpaces",
    "Derived_Variable",
    "Fortran",
    "HDF5_VOL",
    "IME",
    "Kokkos",
    "KVCACHE",
    "LIBPRESSIO",
    "MGARD",
    "MHS",
    "OpenSSL",
    "PNG",
    "PRODM",
    "Python",
    "Sodium",
    "SST",
    "SZ",
    "SZ3",
    "UCX",
    "XRootD",
    "ZeroMQ",
    "ZFP",
)


class Build(NamedTuple):
    """What a built ADIOS2 prefix says about itself.

    Attributes:
        version: The release, from ``ADIOSConfig.h``.
        defines: The names that header defines, which is where ADIOS2 records
            every feature the configure run resolved.
        soname: ``DT_SONAME`` of the installed MPI C++ library.
        needed: Its ``DT_NEEDED`` entries, where the binding to the prefix's
            MPI and parallel HDF5 shows up.
    """

    version: str | None
    defines: frozenset[str]
    soname: str | None
    needed: frozenset[str]


def series(version: str = ADIOS2_VERSION) -> str:
    """Return the ``major.minor`` series a release belongs to.

    Args:
        version: An ADIOS2 release, such as ``2.12.1``.

    Returns:
        The series, such as ``2.12`` — the granularity ADIOS2 versions its
        shared libraries at.
    """
    major, _, rest = version.partition(".")
    return f"{major}.{rest.partition('.')[0]}"


def source_url(version: str = ADIOS2_VERSION) -> str:
    """Return the download URL of the pinned ADIOS2 source tarball."""
    return f"https://github.com/ornladios/ADIOS2/archive/v{version}.tar.gz"


def source_dir_name(version: str = ADIOS2_VERSION) -> str:
    """Return the directory name the source tarball unpacks to.

    The GitHub archive is named after the tag but unpacks to the repository
    name and the version without its ``v``, so the two spellings differ and
    the build script needs this one.

    Args:
        version: The ADIOS2 release being built.

    Returns:
        The directory name, such as ``ADIOS2-2.12.1``.
    """
    return f"ADIOS2-{version}"


def soname(version: str = ADIOS2_VERSION) -> str:
    """Return the soname the installed MPI C++ library declares.

    Args:
        version: The ADIOS2 release being built.

    Returns:
        The soname, such as ``libadios2_cxx_mpi.so.2.12``. ADIOS2 versions
        its libraries by ``major.minor``, as PETSc and SLEPc do.
    """
    return f"{CXX_LIBRARY}.so.{series(version)}"


def feature_define(feature: str) -> str:
    """Return the ``ADIOSConfig.h`` macro that records one feature.

    ADIOS2's CMake takes ``ADIOS2_USE_<Feature>`` in the option's own spelling
    and writes ``ADIOS2_HAVE_<FEATURE>`` in upper case, so the two names are
    not interchangeable and the mapping lives here rather than in two lists.

    Args:
        feature: The feature name as the CMake option spells it.

    Returns:
        The macro name the installed header would define.
    """
    return f"ADIOS2_HAVE_{feature.upper()}"


def configure_arguments(
    *, source_dir: Path, build_dir: Path, prefix: Path, mpi_prefix: Path
) -> list[str]:
    """Return the ADIOS2 CMake configure command.

    Args:
        source_dir: Unpacked ADIOS2 source tree.
        build_dir: Out-of-tree build directory. ADIOS2 refuses an in-source
            build, and a directory of its own is what a warm cache keeps.
        prefix: Shared install prefix.
        mpi_prefix: Prefix holding the MPI compiler wrappers, which is that
            same shared prefix.

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
        # GNUInstallDirs would choose lib64 on the RHEL-family image the
        # manylinux container is built from. wheelbuild.prefix makes that a
        # symlink to lib, but naming the directory keeps the install where
        # every other stage writes rather than relying on the alias.
        "-DCMAKE_INSTALL_LIBDIR=lib",
        # Release, and no -march: the wheel is manylinux_2_34_x86_64 and has
        # to run on every x86-64 machine, not the builder's.
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=ON",
        # The wrappers are MPICH's, from the same prefix: ADIOS2's MPI has to
        # be the one libpetsc and libdolfinx are compiled against.
        f"-DCMAKE_C_COMPILER={mpi_bin / 'mpicc'}",
        f"-DCMAKE_CXX_COMPILER={mpi_bin / 'mpicxx'}",
        # The parallel HDF5 PETSc's configure built, in this same prefix.
        f"-DHDF5_ROOT={prefix}",
        "-DHDF5_PREFER_PARALLEL=ON",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        # Nothing in the wheel runs ADIOS2's own tests or examples, and its
        # install test would run an MPI program at install time.
        "-DBUILD_TESTING=OFF",
        "-DADIOS2_BUILD_EXAMPLES=OFF",
        "-DADIOS2_RUN_INSTALL_TEST=OFF",
        *(f"-DADIOS2_USE_{feature}=ON" for feature in ENABLED_FEATURES),
        *(f"-DADIOS2_USE_{feature}=OFF" for feature in DISABLED_FEATURES),
    ]


def build_arguments(build_dir: Path, jobs: int) -> list[str]:
    """Return the command that compiles a configured ADIOS2.

    Args:
        build_dir: The configured build directory.
        jobs: Parallel build jobs.

    Returns:
        The ``cmake --build`` argument vector.
    """
    return ["cmake", "--build", str(build_dir), "--parallel", str(jobs)]


def install_arguments(build_dir: Path) -> list[str]:
    """Return the command that installs a built ADIOS2.

    Args:
        build_dir: The built build directory.

    Returns:
        The ``cmake --install`` argument vector.
    """
    return ["cmake", "--install", str(build_dir)]


def missing_artefacts(prefix: Path) -> list[Path]:
    """Return the required artefacts an install prefix does not have.

    Args:
        prefix: ADIOS2 install prefix.

    Returns:
        The missing paths, relative to the prefix, in declaration order.
    """
    return [
        relative for relative in REQUIRED_ARTEFACTS if not (prefix / relative).exists()
    ]


def version_in_header(header_text: str) -> str | None:
    """Return the release an installed ``ADIOSConfig.h`` describes.

    Args:
        header_text: Contents of the installed
            ``include/adios2/common/ADIOSConfig.h``.

    Returns:
        The release, such as ``2.12.1``, or ``None`` when the header does not
        carry all three version components.
    """
    defines = macros.defined_macros(header_text)
    components = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        value = defines.get(f"ADIOS2_VERSION_{component}")
        if value is None:
            return None
        components.append(value)
    return ".".join(components)


def observe(prefix: Path) -> Build:
    """Read what a built ADIOS2 prefix says about itself.

    Args:
        prefix: ADIOS2 install prefix.

    Returns:
        The facts :func:`build_problem` judges.

    Raises:
        FileNotFoundError: When ``ADIOSConfig.h`` is missing.
        subprocess.CalledProcessError: When the ELF tools cannot read the
            installed library.
    """
    config = prefix / "include" / "adios2" / "common" / "ADIOSConfig.h"
    library_soname, needed = elf.read_dynamic(prefix / "lib" / f"{CXX_LIBRARY}.so")
    return Build(
        version=version_in_header(config.read_text(encoding="utf-8")),
        defines=macros.read_defined_names(config),
        soname=library_soname,
        needed=needed,
    )


def build_problem(
    build: Build, *, expected_version: str = ADIOS2_VERSION
) -> str | None:
    """Report an ADIOS2 that is not the one this wheel is built around.

    Args:
        build: What the install says about itself.
        expected_version: The release this driver is pinned to.

    Returns:
        A message naming the first thing that is wrong, or ``None``.
    """
    if build.version is None:
        return (
            "the installed ADIOSConfig.h carries no ADIOS2_VERSION_MAJOR/"
            "MINOR/PATCH, so nothing here can confirm which ADIOS2 this "
            "prefix holds."
        )
    if build.version != expected_version:
        return (
            f"this prefix holds ADIOS2 {build.version}, but the wheel is "
            f"built against ADIOS2 {expected_version}. A stale library left "
            "in a warm build cache carries the same soname as the new one, "
            "so the release is checked rather than assumed."
        )

    expected_soname = soname(expected_version)
    if build.soname != expected_soname:
        return (
            f"the installed ADIOS2 C++ library declares the soname "
            f"{build.soname!r} rather than {expected_soname}, so libdolfinx "
            "would be asking the loader for a library the wheel does not "
            "carry."
        )

    for feature in ENABLED_FEATURES:
        if feature_define(feature) not in build.defines:
            return (
                f"this ADIOS2 was configured without {feature} "
                f"(ADIOSConfig.h does not define {feature_define(feature)}). "
                "DOLFINx asks for the CXX and MPI components against the "
                "parallel HDF5 in this prefix, and an ADIOS2 missing either "
                "is I/O the wheel has quietly lost (spec §2)."
            )

    enabled = [
        feature
        for feature in DISABLED_FEATURES
        if feature_define(feature) in build.defines
    ]
    if enabled:
        return (
            f"this ADIOS2 was configured with {', '.join(enabled)}, which "
            "this build turns off by name. Every ADIOS2 feature defaults to "
            "AUTO, so one that is on was found on the build image rather "
            "than chosen — and it is a library the wheel would ship without "
            "a harvested licence (spec §7)."
        )
    return None


def validate(prefix: Path, build: Build) -> Path:
    """Check a built ADIOS2 against everything the wheel needs to be true.

    Args:
        prefix: ADIOS2 install prefix.
        build: What the install says about itself, from :func:`observe`.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is incomplete.
        ValueError: When the build is not the ADIOS2 this wheel is built
            around.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete ADIOS2 install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )

    problem = build_problem(build)
    if problem is not None:
        raise ValueError(problem)
    return prefix


def validate_install(prefix: Path) -> Path:
    """Read the facts about a built ADIOS2, then :func:`validate` them.

    Args:
        prefix: ADIOS2 install prefix.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is incomplete.
        ValueError: When a check fails; see :func:`validate`.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete ADIOS2 install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )
    return validate(prefix, observe(prefix))


def run(*, source_dir: Path, build_dir: Path, prefix: Path, jobs: int) -> Path:
    """Configure, build, install and validate ADIOS2.

    Args:
        source_dir: Unpacked ADIOS2 source tree.
        build_dir: Out-of-tree build directory.
        prefix: Shared install prefix, holding the MPI wrappers and the
            parallel HDF5 to build against.
        jobs: Parallel build jobs.

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
    return validate_install(prefix)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the ADIOS2 build and its validation."""
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
            prefix = validate_install(args.prefix)
        else:
            prefix = run(
                source_dir=args.source_dir,
                build_dir=args.build_dir,
                prefix=args.prefix,
                jobs=args.jobs,
            )
    except (FileNotFoundError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"ADIOS2 {ADIOS2_VERSION} installed into {prefix} with "
        f"{', '.join(ENABLED_FEATURES)}, and none of the "
        f"{len(DISABLED_FEATURES)} optional features this wheel does not ship"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
