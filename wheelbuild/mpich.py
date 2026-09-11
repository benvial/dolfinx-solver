"""Build the Fortran half of MPICH, which is all of MPI this wheel vendors.

The runtime ``libmpi`` comes from the PyPI ``mpich`` wheel and is shared with
mpi4py inside one process (spec §5, ADR-0002). That wheel is C-only: no
``libmpifort``, no ``mpif.h``, no ``mpi.mod``. PETSc's Fortran packages —
MUMPS, ScaLAPACK, SuperLU_DIST — need the headers to compile and the library
to run, so MPICH is compiled here with Fortran enabled, its ``libmpi`` is
thrown away, and only the Fortran half is kept.

That split is the whole risk of this step. The two halves are built by
different people at different times and bound together on the user's machine
by a soname, ``libmpi.so.12``, that MPICH freezes across major series — so the
loader will pair a ``libmpifort`` from one series with a ``libmpi`` from
another without complaint, and the mismatch surfaces as an undefined-symbol
error on first import. :func:`validate` is where that is caught instead: it
compares the symbols our ``libmpifort`` references against the ones the PyPI
wheel's ``libmpi`` exports, which is the check that actually proves the
``mpich>=5.0,<6`` pin (ADR-0001). ``wheelbuild.pin_check`` holds the other
end, that the declared pin still names the series built here.

Every fact the checks work from is read out of the installed files —
``include/mpi.h`` for the release, the ELF headers for the linkage — and never
by running a binary out of the prefix and asking it. A built ``mpichversion``
reports whichever ``libmpi`` its own loader resolves, so on a machine that
already has an MPI installed it cheerfully describes that one instead: the
very confusion this wheel is built to prevent, arriving inside the check meant
to prove it did not happen. Facts from files cannot be shadowed that way, and
they keep the policy testable without a multi-hour build.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from wheelbuild import elf
from wheelbuild._process import check_call

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: MPICH release to build: the one the PyPI ``mpich`` wheel ships, so the
#: Fortran half and the runtime ``libmpi`` come from the same source tree.
#: Moving this means moving the pin in ``pyproject.toml`` with it, which
#: ``wheelbuild.pin_check`` enforces.
MPICH_VERSION = "5.0.1"

#: Soname of the runtime ``libmpi``, frozen across MPICH's 4.x and 5.x series.
#: It is what every vendored library's ``DT_NEEDED`` names and what mpi4py
#: dlopens out of the environment prefix.
MPI_SONAME = "libmpi.so.12"

#: The Fortran library that is the point of this whole step.
FORTRAN_SONAME = "libmpifort.so.12"

#: The PyPI wheel's own ``configure`` line, as ``mpichversion`` reports it
#: (ADR-0001). Our build adds Fortran to it and changes nothing else: a
#: different device or process manager is a different ``libmpi``, and the
#: Fortran half would be built against internals the runtime does not have.
WHEEL_CONFIGURE = ("--with-device=ch4:ucx,ofi", "--with-pm=hydra:gforker")

#: What a usable install has to contain. ``libmpi.so.12`` and ``mpi.h`` are
#: built and read but not vendored — they are how the build proves it produced
#: the same MPICH the PyPI wheel ships.
REQUIRED_ARTEFACTS = (
    Path("include/mpi.h"),
    Path("include/mpi.mod"),
    Path("include/mpif.h"),
    Path(f"lib/{FORTRAN_SONAME}"),
    Path("lib/libmpifort.so"),
    Path(f"lib/{MPI_SONAME}"),
)

#: Where F08 bindings would show up if they were built. The MPICH ABI covers
#: the ``mpif.h`` interface; the ``use mpi_f08`` handle types are derived
#: types whose layout is not part of it, so a stack compiled against them
#: would be bound to this exact build rather than to the series (spec §5).
F08_PATTERNS = ("include/*f08*.mod", "include/*f08*.h", "lib/*f08*")

#: What the wheel keeps out of the install prefix. Deliberately no
#: ``libmpi``: a grafted copy would be a second MPI runtime in a process that
#: already has mpi4py's (spec §5).
VENDORED_ARTEFACTS = (
    Path(f"lib/{FORTRAN_SONAME}"),
    Path("lib/libmpifort.so"),
    Path("include/mpif.h"),
    Path("include/mpi.mod"),
)

#: ``mpi.h`` records the release it was generated for, which is the one fact
#: about the build that no loader can shadow.
_VERSION_DEFINE = re.compile(
    r'^#define\s+MPICH_VERSION\s+"(?P<version>[^"]+)"', re.MULTILINE
)


class Build(NamedTuple):
    """What a built MPICH prefix says about itself.

    Attributes:
        version: The release, from ``include/mpi.h``.
        mpi_soname: ``DT_SONAME`` of the ``libmpi`` this build produced.
        fortran_needed: ``DT_NEEDED`` entries of the vendored ``libmpifort``.
        fortran_undefined: Symbols that ``libmpifort`` references and does not
            define, which the runtime ``libmpi`` will have to supply.
    """

    version: str | None
    mpi_soname: str | None
    fortran_needed: frozenset[str]
    fortran_undefined: frozenset[str]


def series(version: str) -> str:
    """Return the MPICH major series a release belongs to.

    Args:
        version: An MPICH release, such as ``5.0.1``.

    Returns:
        The major version, such as ``5``.
    """
    return version.partition(".")[0]


def source_url(version: str = MPICH_VERSION) -> str:
    """Return the download URL of the pinned MPICH source tarball."""
    return f"https://www.mpich.org/static/downloads/{version}/mpich-{version}.tar.gz"


def configure_arguments(*, source_dir: Path, prefix: Path) -> list[str]:
    """Return the MPICH ``configure`` command.

    The wheel's own line (:data:`WHEEL_CONFIGURE`) plus the Fortran bindings it
    leaves out, minus the F08 ones. Nothing else is varied, because the
    ``libmpi`` this configuration produces has to be the one the Fortran half
    will meet again at run time.

    Args:
        source_dir: Unpacked MPICH source tree.
        prefix: Install prefix for the build.

    Returns:
        The ``configure`` argument vector.
    """
    return [
        str(source_dir / "configure"),
        f"--prefix={prefix}",
        *WHEEL_CONFIGURE,
        # Only the shared libraries are vendored, and the static halves double
        # the compile of a tree that is already the slow part of a cold build.
        "--enable-shared",
        "--disable-static",
        # The Fortran bindings the PyPI wheel does not ship: mpif.h (f77) for
        # PETSc's Fortran packages, mpi.mod (f90) for the ones that prefer the
        # module, and not the f08 bindings, whose types the ABI does not cover.
        "--disable-f08",
    ]


def missing_artefacts(prefix: Path) -> list[Path]:
    """Return the required artefacts an install prefix does not have.

    Args:
        prefix: MPICH install prefix.

    Returns:
        The missing paths, relative to the prefix, in declaration order.
    """
    return [
        relative for relative in REQUIRED_ARTEFACTS if not (prefix / relative).exists()
    ]


def f08_artefacts(prefix: Path) -> list[Path]:
    """Return any F08 binding artefacts an install prefix has.

    Args:
        prefix: MPICH install prefix.

    Returns:
        The paths found, relative to the prefix, sorted.
    """
    found = {
        path.relative_to(prefix)
        for pattern in F08_PATTERNS
        for path in prefix.glob(pattern)
    }
    return sorted(found)


def version_in_header(header_text: str) -> str | None:
    """Return the MPICH release an ``mpi.h`` was generated for.

    Args:
        header_text: Contents of the installed ``include/mpi.h``.

    Returns:
        The release, or ``None`` when the header declares none.
    """
    match = _VERSION_DEFINE.search(header_text)
    return None if match is None else match["version"]


def observe(prefix: Path) -> Build:
    """Read what a built MPICH prefix says about itself.

    Args:
        prefix: MPICH install prefix.

    Returns:
        The facts :func:`build_problem` and :func:`interop_problem` judge.

    Raises:
        subprocess.CalledProcessError: When the ELF tools cannot read a
            library.
    """
    fortran_library = prefix / "lib" / FORTRAN_SONAME
    mpi_soname, _ = elf.read_dynamic(prefix / "lib" / MPI_SONAME)
    _, fortran_needed = elf.read_dynamic(fortran_library)
    _, fortran_undefined = elf.read_symbols(fortran_library)
    return Build(
        version=version_in_header(
            (prefix / "include" / "mpi.h").read_text(encoding="utf-8")
        ),
        mpi_soname=mpi_soname,
        fortran_needed=fortran_needed,
        fortran_undefined=frozenset(fortran_undefined),
    )


def build_problem(build: Build, *, expected_version: str = MPICH_VERSION) -> str | None:
    """Report a build that is not the MPICH the runtime ``libmpi`` comes from.

    Args:
        build: What the install says about itself.
        expected_version: The release this driver is pinned to.

    Returns:
        A message, or ``None`` when the build matches on every count.
    """
    if build.version is None:
        return (
            "the installed mpi.h declares no MPICH_VERSION, so nothing here "
            "can confirm which MPICH the Fortran half was built from."
        )
    if build.version != expected_version:
        return (
            f"this build is MPICH {build.version}, but the wheel is built "
            f"against MPICH {expected_version} — the release the PyPI mpich "
            "wheel ships. A libmpifort from one series and a libmpi from "
            "another share the soname and bind happily, then fail at the "
            "user's first import. Move wheelbuild.mpich.MPICH_VERSION and the "
            "mpich pin in pyproject.toml together (ADR-0001)."
        )

    if build.mpi_soname != MPI_SONAME:
        return (
            f"this build's libmpi declares the soname {build.mpi_soname!r} "
            f"rather than {MPI_SONAME}. Every vendored library names that "
            "soname in its DT_NEEDED and mpi4py dlopens it from the "
            "environment prefix, so a different one is a stack the PyPI "
            "wheel's libmpi can never satisfy."
        )

    if MPI_SONAME not in build.fortran_needed:
        return (
            f"the vendored {FORTRAN_SONAME} does not ask the loader for "
            f"{MPI_SONAME} (it needs "
            f"{', '.join(sorted(build.fortran_needed)) or 'nothing'}). The "
            "Fortran half is only usable because it binds the C half by that "
            "soname, which is how it reaches the libmpi mpi4py already "
            "loaded (ADR-0002)."
        )
    return None


def interop_problem(
    fortran_undefined: Iterable[str],
    runtime_exported: Iterable[str],
) -> str | None:
    """Report symbols the vendored ``libmpifort`` needs and ``libmpi`` lacks.

    This is the check ADR-0001 rests the pin on: not that the version numbers
    look compatible, but that the library a user's ``pip install`` actually
    resolves can satisfy every MPI symbol the Fortran half references.

    Args:
        fortran_undefined: Undefined symbols of the vendored ``libmpifort``.
        runtime_exported: Symbols the PyPI wheel's ``libmpi`` defines.

    Returns:
        A message naming the unsatisfied symbols, or ``None`` when the
        runtime covers all of them.
    """
    needed = elf.mpi_symbols(fortran_undefined)
    missing = sorted(needed - set(runtime_exported))
    if not missing:
        return None
    return (
        f"the vendored {FORTRAN_SONAME} references {len(missing)} MPI "
        f"symbol(s) the PyPI mpich wheel's {MPI_SONAME} does not export: "
        f"{', '.join(missing)}. The soname is the same either way, so the "
        "loader would bind them and fail at the user's first import. Build "
        "the Fortran half from the MPICH release the wheel ships, and move "
        "the mpich pin with it (ADR-0001)."
    )


def validate(
    prefix: Path,
    build: Build,
    runtime_exported: Iterable[str],
) -> Path:
    """Check a built MPICH against everything the wheel needs to be true.

    Args:
        prefix: MPICH install prefix.
        build: What the install says about itself, from :func:`observe`.
        runtime_exported: Symbols the PyPI wheel's ``libmpi`` defines.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is missing something the build or
            the wheel needs.
        ValueError: When the build carries F08 bindings, is not the pinned
            MPICH, or needs symbols the runtime ``libmpi`` does not export.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete MPICH install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )

    f08 = f08_artefacts(prefix)
    if f08:
        carried = ", ".join(str(relative) for relative in f08)
        raise ValueError(
            f"this MPICH build carries F08 bindings ({carried}). The MPICH "
            "ABI covers the mpif.h interface, not the use mpi_f08 derived "
            "types, so anything compiled against them is bound to this build "
            "rather than to the series the wheel pins (spec §5). Configure "
            "with --disable-f08."
        )

    problem = build_problem(build)
    if problem is None:
        problem = interop_problem(build.fortran_undefined, runtime_exported)
    if problem is not None:
        raise ValueError(problem)
    return prefix


def validate_install(prefix: Path, runtime_libmpi: Path) -> Path:
    """Read the facts about a built MPICH, then :func:`validate` them.

    Args:
        prefix: MPICH install prefix.
        runtime_libmpi: The PyPI ``mpich`` wheel's ``libmpi.so.12``, installed
            into the build venv — the library a user's install resolves too.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install, or the runtime library, is
            missing something.
        ValueError: When a check fails; see :func:`validate`.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete MPICH install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )
    if not runtime_libmpi.exists():
        raise FileNotFoundError(
            f"no runtime libmpi at {runtime_libmpi}: the interop check needs "
            "the PyPI mpich wheel installed, since that is the libmpi the "
            "vendored libmpifort meets on a user's machine (ADR-0001)."
        )

    runtime_exported, _ = elf.read_symbols(runtime_libmpi)
    return validate(prefix, observe(prefix), runtime_exported)


def run(
    *,
    source_dir: Path,
    build_dir: Path,
    prefix: Path,
    runtime_libmpi: Path,
    jobs: int,
) -> Path:
    """Configure, build, install and validate the Fortran half of MPICH.

    The ``libmpi`` this produces stays in the prefix — PETSc and DOLFINx link
    against it during the build, and ``auditwheel repair --exclude`` keeps it
    out of the wheel (spec §5) — but it is never vendored.

    Args:
        source_dir: Unpacked MPICH source tree.
        build_dir: Out-of-tree build directory; cache it to skip rebuilds.
        prefix: Shared install prefix.
        runtime_libmpi: The PyPI ``mpich`` wheel's ``libmpi.so.12``.
        jobs: Parallel build jobs.

    Returns:
        The validated install prefix.
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    check_call(configure_arguments(source_dir=source_dir, prefix=prefix), cwd=build_dir)
    check_call(["make", f"-j{jobs}"], cwd=build_dir)
    check_call(["make", "install"], cwd=build_dir)
    return validate_install(prefix, runtime_libmpi)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the MPICH build and its validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument(
        "--runtime-libmpi",
        type=Path,
        required=True,
        help=f"the PyPI mpich wheel's {MPI_SONAME}, for the interop check",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check an already-built prefix instead of building one",
    )
    # Not needed by --validate-only, which is how a warm build cache re-proves
    # a prefix it did not have to rebuild.
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.validate_only and (args.source_dir is None or args.build_dir is None):
        parser.error("--source-dir and --build-dir are required to build")

    try:
        if args.validate_only:
            prefix = validate_install(args.prefix, args.runtime_libmpi)
        else:
            prefix = run(
                source_dir=args.source_dir,
                build_dir=args.build_dir,
                prefix=args.prefix,
                runtime_libmpi=args.runtime_libmpi,
                jobs=args.jobs,
            )
    except (FileNotFoundError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"MPICH {MPICH_VERSION} Fortran bindings installed into {prefix}; "
        f"every MPI symbol {FORTRAN_SONAME} needs is exported by "
        f"{args.runtime_libmpi}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
