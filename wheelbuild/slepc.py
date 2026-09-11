"""Build SLEPc against the PETSc this wheel just vendored.

SLEPc is the eigensolver half of the stack, and it is not a package that can
be pinned independently: a SLEPc release is written against one PETSc minor
series, reads that PETSc's installed configuration at configure time, and
links the resulting ``libslepc`` against that exact ``libpetsc``. Complex
scalars, index width and precision all come across from PETSc rather than
being chosen here, which is why this driver has a configure line of four
options where :mod:`wheelbuild.petsc` has forty.

What it does have to prove is the coupling. ``libslepc`` names ``libpetsc``
by soname, and a soname is satisfied by any library with that name — so a
SLEPc left over in a warm build cache from a previous PETSc would link
happily against the new one and fail somewhere inside a solve. The checks
here read the installed ``slepcversion.h`` and ``libslepc``'s dynamic section
and refuse anything that is not this release bound to this PETSc, on the same
principle the MPICH driver established: every fact comes from an installed
file, never from running a binary out of the prefix.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from wheelbuild import elf, macros, petsc
from wheelbuild._process import check_call

if TYPE_CHECKING:
    from collections.abc import Sequence

#: SLEPc release to build. Its minor series has to be PETSc's:
#: :func:`coupling_problem` refuses the pair when they drift apart, so this
#: number and :data:`wheelbuild.petsc.PETSC_VERSION` move together.
SLEPC_VERSION = "3.25.1"

#: What a finished prefix install has to contain.
REQUIRED_ARTEFACTS = (
    Path("include/slepc.h"),
    Path("include/slepcconf.h"),
    Path("include/slepcversion.h"),
    Path("lib/libslepc.so"),
)

#: The eigensolver packages SLEPc can pull in and this wheel does not ship.
#: They are off by default; naming them keeps that a decision rather than a
#: default, in the same spirit as PETSc's line (spec §8). slepc4py is built
#: separately, against the finished install, in the ticket that follows.
DISABLED_PACKAGES = ("arpack", "blopex", "primme", "slepc4py")


class Build(NamedTuple):
    """What a built SLEPc prefix says about itself.

    Attributes:
        version: The release, from ``include/slepcversion.h``.
        soname: ``DT_SONAME`` of the installed ``libslepc``.
        needed: ``DT_NEEDED`` entries of the installed ``libslepc``, which is
            where the binding to one ``libpetsc`` shows up.
    """

    version: str | None
    soname: str | None
    needed: frozenset[str]


def series(version: str) -> str:
    """Return the minor series a SLEPc release belongs to.

    Args:
        version: A SLEPc release, such as ``3.25.1``.

    Returns:
        The ``major.minor`` series, such as ``3.25``.
    """
    return petsc.series(version)


def source_url(version: str = SLEPC_VERSION) -> str:
    """Return the download URL of the pinned SLEPc source tarball."""
    return f"https://slepc.upv.es/download/distrib/slepc-{version}.tar.gz"


def soname(version: str = SLEPC_VERSION) -> str:
    """Return the soname the built ``libslepc`` declares.

    Args:
        version: The SLEPc release being built.

    Returns:
        The soname, such as ``libslepc.so.3.25`` — versioned by minor series,
        as PETSc's is.
    """
    return f"libslepc.so.{series(version)}"


def configure_arguments(*, source_dir: Path, prefix: Path) -> list[str]:
    """Return the SLEPc ``configure`` command.

    Everything about the resulting library that the wheel cares about —
    scalar type, precision, index width, which solvers PETSc brought — is read
    out of the PETSc install this is pointed at, through ``PETSC_DIR`` in the
    environment. What is left is where to install and what not to pull in.

    Args:
        source_dir: Unpacked SLEPc source tree.
        prefix: Shared install prefix, the same one PETSc installed into.

    Returns:
        The ``configure`` argument vector.
    """
    return [
        str(source_dir / "configure"),
        f"--prefix={prefix}",
        *(f"--with-{package}=0" for package in DISABLED_PACKAGES),
    ]


def build_environment(
    *, source_dir: Path, petsc_prefix: Path, environ: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the environment SLEPc's configure and make need.

    SLEPc takes its PETSc from the environment, not from a flag. ``PETSC_ARCH``
    is set to the empty string on purpose rather than left alone: a
    prefix-installed PETSc has no arch directory, and an inherited
    ``PETSC_ARCH`` from some other build would send SLEPc looking inside one.

    Args:
        source_dir: Unpacked SLEPc source tree.
        petsc_prefix: Prefix the vendored PETSc is installed in.
        environ: Environment to extend. Defaults to this process's.

    Returns:
        A complete environment for the SLEPc build.
    """
    return {
        **(os.environ if environ is None else environ),
        "SLEPC_DIR": str(source_dir),
        "PETSC_DIR": str(petsc_prefix),
        "PETSC_ARCH": "",
    }


def arch_in_variables(variables_text: str) -> str | None:
    """Return the ``PETSC_ARCH`` SLEPc's configure recorded for its build tree.

    Args:
        variables_text: Contents of ``lib/slepc/conf/slepcvariables``.

    Returns:
        The arch name, or ``None`` when the file names none.
    """
    for line in variables_text.splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "PETSC_ARCH":
            return value.strip()
    return None


def configured_arch(source_dir: Path) -> str:
    """Return the build directory name SLEPc's configure chose.

    SLEPc does not build a prefix install in place. It derives a name from the
    PETSc it configured against — ``installed-arch-linux2-c-opt-complex`` for
    ours — creates that directory inside the source tree, and records the name
    in ``lib/slepc/conf/slepcvariables`` for its makefiles to pick up. So
    ``make`` has to be told the same name: passing an empty ``PETSC_ARCH`` on
    the command line looks like the right thing for a prefix install and is
    not, because a command-line variable overrides the value configure
    recorded, and the build then looks for its own rules in a directory that
    does not exist.

    Args:
        source_dir: Unpacked SLEPc source tree, already configured.

    Returns:
        The arch name to pass to ``make``.

    Raises:
        FileNotFoundError: When configure has not written the file.
        ValueError: When the file names no arch.
    """
    variables = source_dir / "lib" / "slepc" / "conf" / "slepcvariables"
    if not variables.exists():
        raise FileNotFoundError(
            f"no {variables}: SLEPc's configure writes this file, and the "
            "build directory it names is where the rules make needs live."
        )
    arch = arch_in_variables(variables.read_text(encoding="utf-8"))
    if not arch:
        raise ValueError(
            f"{variables} names no PETSC_ARCH, so there is no way to tell "
            "make which of SLEPc's build directories this configure produced."
        )
    return arch


def missing_artefacts(prefix: Path) -> list[Path]:
    """Return the required artefacts an install prefix does not have.

    Args:
        prefix: SLEPc install prefix.

    Returns:
        The missing paths, relative to the prefix, in declaration order.
    """
    return [
        relative for relative in REQUIRED_ARTEFACTS if not (prefix / relative).exists()
    ]


def version_in_header(header_text: str) -> str | None:
    """Return the release an installed ``slepcversion.h`` describes.

    Args:
        header_text: Contents of the installed ``include/slepcversion.h``.

    Returns:
        The release, such as ``3.25.1``, or ``None`` when the header does not
        carry all three version components.
    """
    defines = macros.defined_macros(header_text)
    components = []
    for component in ("MAJOR", "MINOR", "SUBMINOR"):
        value = defines.get(f"SLEPC_VERSION_{component}")
        if value is None:
            return None
        components.append(value)
    return ".".join(components)


def coupling_problem(
    slepc_version: str = SLEPC_VERSION, petsc_version: str = petsc.PETSC_VERSION
) -> str | None:
    """Report a SLEPc release that is not written against our PETSc.

    This needs nothing but the two pins in this repository, so :func:`run`
    asks it before doing anything else: a mismatched pair is a mistake worth
    catching in the second before the build rather than in the middle of it.

    Args:
        slepc_version: The SLEPc release this driver builds.
        petsc_version: The PETSc release :mod:`wheelbuild.petsc` builds.

    Returns:
        A message, or ``None`` when the two name the same minor series.
    """
    if series(slepc_version) == petsc.series(petsc_version):
        return None
    return (
        f"SLEPc {slepc_version} and PETSc {petsc_version} are different minor "
        f"series ({series(slepc_version)} against "
        f"{petsc.series(petsc_version)}). A SLEPc release is written against "
        "one PETSc series and refuses to configure against another, so the "
        "two version pins in wheelbuild move together (spec §6)."
    )


def observe(prefix: Path) -> Build:
    """Read what a built SLEPc prefix says about itself.

    Args:
        prefix: SLEPc install prefix.

    Returns:
        The facts :func:`build_problem` judges.

    Raises:
        FileNotFoundError: When ``slepcversion.h`` is missing.
        subprocess.CalledProcessError: When the ELF tools cannot read
            ``libslepc``.
    """
    library_soname, needed = elf.read_dynamic(prefix / "lib" / "libslepc.so")
    return Build(
        version=version_in_header(
            (prefix / "include" / "slepcversion.h").read_text(encoding="utf-8")
        ),
        soname=library_soname,
        needed=needed,
    )


def build_problem(
    build: Build,
    *,
    expected_version: str = SLEPC_VERSION,
    petsc_version: str = petsc.PETSC_VERSION,
) -> str | None:
    """Report a SLEPc that is not this release bound to our PETSc.

    Args:
        build: What the install says about itself.
        expected_version: The SLEPc release this driver is pinned to.
        petsc_version: The PETSc release the wheel vendors.

    Returns:
        A message naming the first thing that is wrong, or ``None``.
    """
    if build.version is None:
        return (
            "the installed slepcversion.h carries no SLEPC_VERSION_MAJOR/"
            "MINOR/SUBMINOR, so nothing here can confirm which SLEPc this "
            "prefix holds."
        )
    if build.version != expected_version:
        return (
            f"this prefix holds SLEPc {build.version}, but the wheel is built "
            f"against SLEPc {expected_version}. A stale libslepc in a warm "
            "build cache links against the new libpetsc by soname without "
            "complaint, which is why the release is checked rather than "
            "assumed."
        )

    expected_soname = soname(expected_version)
    if build.soname != expected_soname:
        return (
            f"the installed libslepc declares the soname {build.soname!r} "
            f"rather than {expected_soname}, so slepc4py and anything else in "
            "the wheel would be asking the loader for a library that is not "
            "there."
        )

    petsc_soname = petsc.soname(petsc_version)
    if petsc_soname not in build.needed:
        return (
            f"the installed libslepc does not ask the loader for "
            f"{petsc_soname} (it needs "
            f"{', '.join(sorted(build.needed)) or 'nothing'}). SLEPc that is "
            "not bound to the PETSc this build vendored is SLEPc built "
            "against something else that was on the machine."
        )
    return None


def validate(prefix: Path, build: Build) -> Path:
    """Check a built SLEPc against everything the wheel needs to be true.

    Args:
        prefix: SLEPc install prefix.
        build: What the install says about itself, from :func:`observe`.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is incomplete, or when the
            ``libpetsc`` it binds is not the one in this prefix.
        ValueError: When the build is not this SLEPc bound to our PETSc.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete SLEPc install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )

    problem = coupling_problem()
    if problem is None:
        problem = build_problem(build)
    if problem is not None:
        raise ValueError(problem)

    vendored_petsc = prefix / "lib" / petsc.soname()
    if not vendored_petsc.exists():
        raise FileNotFoundError(
            f"libslepc asks for {petsc.soname()} and this prefix does not "
            f"contain it at {vendored_petsc}. The soname alone would be "
            "satisfied by any PETSc on the machine; the wheel ships one, and "
            "it has to be this one."
        )
    return prefix


def validate_install(prefix: Path) -> Path:
    """Read the facts about a built SLEPc, then :func:`validate` them.

    Args:
        prefix: SLEPc install prefix.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is incomplete.
        ValueError: When a check fails; see :func:`validate`.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete SLEPc install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )
    return validate(prefix, observe(prefix))


def run(*, source_dir: Path, prefix: Path, jobs: int) -> Path:
    """Configure, build, install and validate SLEPc against the vendored PETSc.

    Args:
        source_dir: Unpacked SLEPc source tree, which is also its build tree.
        prefix: Shared install prefix, holding the PETSc to build against.
        jobs: Parallel build jobs.

    Returns:
        The validated install prefix.

    Raises:
        ValueError: When the pinned SLEPc and PETSc are different series.
    """
    problem = coupling_problem()
    if problem is not None:
        raise ValueError(problem)

    environment = build_environment(source_dir=source_dir, petsc_prefix=prefix)
    check_call(
        configure_arguments(source_dir=source_dir, prefix=prefix),
        cwd=source_dir,
        env=environment,
    )
    make = [
        "make",
        f"SLEPC_DIR={source_dir}",
        f"PETSC_DIR={prefix}",
        f"PETSC_ARCH={configured_arch(source_dir)}",
        f"MAKE_NP={jobs}",
    ]
    check_call(make, cwd=source_dir, env=environment)
    check_call([*make, "install"], cwd=source_dir, env=environment)
    return validate_install(prefix)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the SLEPc build and its validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check an already-built prefix instead of building one",
    )
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.validate_only and args.source_dir is None:
        parser.error("--source-dir is required to build")

    try:
        if args.validate_only:
            prefix = validate_install(args.prefix)
        else:
            prefix = run(source_dir=args.source_dir, prefix=args.prefix, jobs=args.jobs)
    except (FileNotFoundError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"SLEPc {SLEPC_VERSION} installed into {prefix}, linked against the "
        f"vendored {petsc.soname()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
