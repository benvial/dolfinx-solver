"""Build DOLFINx itself: the C++ core, then its nanobind Python bindings.

This is the product. Everything the earlier stages installed into the shared
prefix — the Fortran half of MPICH, complex PETSc with its whole
``--download-*`` tree, SLEPc, our petsc4py and slepc4py, ADIOS2, KaHIP — exists
so that these two builds can happen against it and produce a ``dolfinx``
package that imports.

The two halves are built differently and checked differently:

* **The C++ core** is a plain CMake build into the shared prefix. Its flag set
  is maximal and explicit (spec §8): every optional package DOLFINx knows
  about is named ``ON`` or ``OFF`` rather than left to its default, because
  DOLFINx defaults all of them to ``ON`` and then downgrades silently to
  "not found" — so a prefix missing ADIOS2 produces a DOLFINx without I/O and
  a log line, not a failure. What the build actually resolved is then read
  back out of the installed ``dolfinx.pc``, whose ``definitions=`` line is
  where CMake records the ``HAS_*`` macros the library was compiled with, and
  out of ``libdolfinx``'s own ``DT_NEEDED`` list.
* **The bindings** are built by scikit-build-core through ``pip wheel``, with
  ``wheel.py-api=cp312``. That one setting is what makes the wheel serve 3.12,
  3.13 and 3.14 alike: it sets ``SKBUILD_SABI_VERSION``, which flips DOLFINx's
  own ``nanobind_add_module(... STABLE_ABI)`` on and, with it, the
  ``MPI4PY_LIMITED_API=1`` and ``CYTHON_COMPILING_IN_LIMITED_API=1``
  definitions the mpi4py and petsc4py casters need (spec §4). The proof is the
  ``cp312-abi3`` tag on the built wheel and the ``cpp.abi3.so`` inside it.

Two things about the staged package are decisions rather than mechanics:

* **The extension's rpath is rewritten**, as petsc4py's and slepc4py's are.
  DOLFINx's bindings sit one directory below the wheel root rather than two,
  so :data:`RELATIVE_RPATH` climbs one level, not two.
* **The ``.dist-info`` is kept**, which is the opposite of what
  :mod:`wheelbuild.bindings` does with petsc4py's. ``dolfinx/__init__.py``
  ends with ``__version__ = version("fenics-dolfinx")``, an
  ``importlib.metadata`` lookup that raises if no distribution of that name is
  installed — so stripping the metadata would make ``import dolfinx`` fail
  outright. Its ``RECORD`` is dropped: without one, ``pip install
  fenics-dolfinx`` refuses to uninstall the copy inside our wheel instead of
  silently replacing our DOLFINx with upstream's PETSc-free build.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from wheelbuild import adios2, bindings, elf, kahip, mpich, petsc, pin_check, slepc
from wheelbuild import version as version_module
from wheelbuild._process import capture, check_call

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The DOLFINx release this wheel ships, as upstream tags it. It is the
#: package's own version module that decides (spec §3), not this driver: the
#: wheel's version *is* the upstream release it mirrors.
DOLFINX_VERSION = version_module.DOLFINX_VERSION

#: That release without a packaging segment, which is what DOLFINx's own
#: ``project(DOLFINX VERSION ...)`` line carries and therefore what the
#: installed ``dolfinx.pc`` reports.
RELEASE = version_module.RELEASE

#: Distribution name of the bindings wheel upstream's ``python/pyproject.toml``
#: builds. We never publish it; see the module docstring for why its
#: ``.dist-info`` nevertheless rides inside ours.
DISTRIBUTION = "fenics_dolfinx"

#: Import name of the package, which is also the directory it stages into.
IMPORT_NAME = "dolfinx"

#: The nanobind extension inside it. The ``abi3`` in the name is the
#: limited-API build's own record of itself.
EXTENSION_PATH = Path(IMPORT_NAME) / "cpp.abi3.so"

#: The nanobind release the bindings are built with, and it is a coupling, not
#: a floor. nanobind gives every extension an ABI tag naming the version of
#: its internal data structures; two extensions share a type registry only
#: when their tags agree, and DOLFINx's bindings hand basix's C++ objects
#: straight to basix's. The ``fenics-basix`` wheel on PyPI carries internals
#: ``v19``, which is nanobind 2.12; built against a newer nanobind our
#: extension gets its own registry and every call that takes a basix element
#: fails with "incompatible function arguments" — at run time, in the user's
#: first ``functionspace`` call, not in the build. :func:`nanobind_problem`
#: is the check that turns that into a build failure; this pin is the fix.
NANOBIND_VERSION = "2.12"

#: That pin as a requirement pip can install.
NANOBIND_REQUIREMENT = f"nanobind=={NANOBIND_VERSION}.*"

#: Where the ``fenics-basix`` wheel keeps its own nanobind extension, relative
#: to the directory it is installed in. Its ABI tag is the one ours has to
#: match.
BASIX_EXTENSION = Path("basix/_basixcpp.abi3.so")

#: Where the ``fenics-basix`` wheel keeps ``libbasix.so``, relative to the
#: directory it is installed in. The trio is depended on and never re-vendored
#: (spec §10), so this is a directory *another* wheel owns — and reaching it
#: by a relative path is what lets ours stay free of a second copy. It works
#: for the same reason the rest of the layout does: both wheels install into
#: one site-packages, which is the venv-style prefix spec §5 already requires.
BASIX_LIBRARY_DIR = Path("basix/lib")

#: The rpath written onto the extension, in search order. It lives one
#: directory down from the wheel root, so it climbs one level to reach both
#: the sibling library directory and basix's — petsc4py's and slepc4py's climb
#: two, because theirs sit in a ``lib`` subdirectory of their own package.
RELATIVE_RPATH_ENTRIES = (
    "$ORIGIN/../" + bindings.VENDORED_LIBRARY_DIR.as_posix(),
    "$ORIGIN/../" + BASIX_LIBRARY_DIR.as_posix(),
)

#: Those entries as the loader reads them.
RELATIVE_RPATH = ":".join(RELATIVE_RPATH_ENTRIES)


def soname(release: str = RELEASE) -> str:
    """Return the soname the built ``libdolfinx`` declares.

    DOLFINx versions its shared library by ``major.minor``, as PETSc and SLEPc
    do, so this name survives a patch release and a stale core left in a warm
    cache would share it with a fresh one. That is what the release check in
    :func:`build_problem` is for; this is what the bindings ask the loader
    for.

    Args:
        release: The DOLFINx release, without a packaging segment.

    Returns:
        The soname, such as ``libdolfinx.so.0.11``.
    """
    major, _, rest = release.partition(".")
    return f"libdolfinx.so.{major}.{rest.partition('.')[0]}"


#: What a finished C++ install has to contain.
REQUIRED_ARTEFACTS = (
    Path("include/dolfinx.h"),
    Path("include/dolfinx/common/version.h"),
    Path("lib/libdolfinx.so"),
    Path("lib/pkgconfig/dolfinx.pc"),
    Path("lib/cmake/dolfinx/DOLFINXConfig.cmake"),
)

#: Optional packages the wheel is built with, and the ``dolfinx.pc`` define
#: that proves each one was found rather than skipped. Spec §2: never trim a
#: feature for packaging convenience, and DOLFINx's "enabled but not found"
#: path is a log line, not an error.
REQUIRED_DEFINES = {
    "HAS_PETSC": "PETSc",
    "HAS_SLEPC": "SLEPc",
    "HAS_ADIOS2": "ADIOS2",
    "HAS_PTSCOTCH": "PT-SCOTCH",
    "HAS_KAHIP": "KaHIP",
    "HAS_SUPERLU_DIST": "SuperLU_DIST",
}

#: Defines that must not appear.
FORBIDDEN_DEFINES = {
    "HAS_PARMETIS": (
        "ParMETIS may not be redistributed without permission from the "
        "University of Minnesota, so a wheel carrying it cannot be published "
        "at all. Parallel partitioning is PT-SCOTCH's and KaHIP's (spec §7)"
    )
}

#: Libraries ``libdolfinx`` has to be linked against, and where the soname it
#: must ask for is read from. Reading the soname off the installed file rather
#: than spelling it here means a version bump in PETSc's ``--download-*`` tree
#: moves this check with it, and means the check is against the copy in *this*
#: prefix rather than the name of some copy.
REQUIRED_LINKAGE = {
    "PETSc": Path("lib/libpetsc.so"),
    "SLEPc": Path("lib/libslepc.so"),
    "ADIOS2": Path(f"lib/{adios2.CXX_LIBRARY}.so"),
    "PT-SCOTCH": Path("lib/libptscotch.so"),
    "KaHIP": Path("lib/libkahip.so"),
    "ParHIP": Path("lib/libparhip_interface.so"),
    "SuperLU_DIST": Path("lib/libsuperlu_dist.so"),
    "parallel HDF5": Path("lib/libhdf5.so"),
}


#: The pure-Python FEniCS packages DOLFINx is built and run against. The C++
#: core compiles against Basix's headers and FFCx's ``ufcx.h``, so the build
#: venv needs the same releases the wheel will depend on at run time — and
#: those are declared once, in our own ``pyproject.toml``, copied verbatim
#: from upstream's (spec §10). Reading them back from there is what keeps the
#: build and the metadata from drifting apart.
UPSTREAM_TRIO = pin_check.UPSTREAM_TRIO


def upstream_trio_requirements(pyproject_text: str | None = None) -> list[str]:
    """Return the requirement strings for the upstream trio, as declared.

    Args:
        pyproject_text: Contents of the project's ``pyproject.toml``. Read
            from the repository when not given.

    Returns:
        The requirements, in :data:`UPSTREAM_TRIO` order.

    Raises:
        LookupError: When one of them is not declared. The C++ build would
            then compile against whatever release pip happened to resolve,
            which is exactly the coupling spec §10 pins down.
    """
    if pyproject_text is None:
        pyproject_text = pin_check.PYPROJECT_PATH.read_text(encoding="utf-8")
    declared = {
        canonicalize_name(Requirement(dependency).name): dependency
        for dependency in tomllib.loads(pyproject_text)
        .get("project", {})
        .get("dependencies", [])
    }
    requirements = []
    for name in UPSTREAM_TRIO:
        requirement = declared.get(canonicalize_name(name))
        if requirement is None:
            raise LookupError(
                f"no {name} dependency is declared in pyproject.toml. DOLFINx "
                "is built against the trio and depends on it at run time; the "
                "pins are upstream's own and are the coupling contract "
                "(spec §10)."
            )
        # Normalised rather than passed through: the build script installs
        # these by word-splitting one line of output, and a requirement
        # printed as written could carry spaces around its specifiers.
        requirements.append(str(Requirement(requirement)))
    return requirements


#: nanobind's ABI tag as it appears in a compiled extension: a domain string
#: naming the internals version, the compiler and the standard library, such
#: as ``v19_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1_stable``. nanobind 3
#: spells the version later in the string than nanobind 2 does, so the pattern
#: brackets the whole thing rather than picking the number out of a fixed
#: place: what matters is whether two extensions carry the same tag.
_NANOBIND_ABI_TAG = re.compile(
    rb"(?<![A-Za-z0-9_])[A-Za-z0-9_]*system_[A-Za-z0-9_]*stable"
)


def nanobind_abi_tag(extension: Path) -> str | None:
    """Return the nanobind ABI tag compiled into an extension module.

    The tag is read out of the binary rather than from the nanobind package
    that happens to be installed, for the reason the rest of this build reads
    installed files: what a wheel on PyPI was built with months ago is
    knowable only from the wheel.

    Args:
        extension: A nanobind extension module.

    Returns:
        The tag, or ``None`` when the file carries none — which means it was
        not built by a nanobind that records one.

    Raises:
        FileNotFoundError: When the extension does not exist.
    """
    match = _NANOBIND_ABI_TAG.search(extension.read_bytes())
    return None if match is None else match.group().decode("ascii")


def nanobind_problem(ours: str | None, basix: str | None) -> str | None:
    """Report bindings that cannot exchange types with the basix wheel's.

    Args:
        ours: The ABI tag of our DOLFINx extension.
        basix: The ABI tag of the installed ``fenics-basix`` extension.

    Returns:
        A message, or ``None`` when the two agree.
    """
    if ours is not None and ours == basix:
        return None
    return (
        f"the DOLFINx bindings carry the nanobind ABI tag {ours!r} and the "
        f"installed fenics-basix extension carries {basix!r}. Two nanobind "
        "extensions share a type registry only when those agree, and DOLFINx "
        "passes basix's C++ elements straight into ours: a mismatch is not a "
        "build error but a TypeError in the user's first functionspace call, "
        f"reported as 'incompatible function arguments'. Build against "
        f"{NANOBIND_REQUIREMENT}, which is the nanobind the published basix "
        "wheel was built with."
    )


class Build(NamedTuple):
    """What a built DOLFINx C++ install says about itself.

    Attributes:
        version: The release, from the installed ``dolfinx.pc``.
        defines: The macros that file records the library as compiled with,
            which is where the ``HAS_*`` features show up.
        runpath: ``libdolfinx``'s search path.
        needed: Its ``DT_NEEDED`` entries.
    """

    version: str | None
    defines: frozenset[str]
    runpath: tuple[str, ...]
    needed: frozenset[str]


class Staged(NamedTuple):
    """What the staged Python extension says about itself.

    Attributes:
        runpath: Its ``DT_RUNPATH`` directories, where the absolute build path
            would show up.
        needed: Its ``DT_NEEDED`` entries, where the binding to the C++ core
            shows up.
    """

    runpath: tuple[str, ...]
    needed: frozenset[str]


def source_url(dolfinx_version: str = DOLFINX_VERSION) -> str:
    """Return the download URL of the pinned DOLFINx source tarball."""
    return (
        f"https://github.com/FEniCS/dolfinx/archive/refs/tags/v{dolfinx_version}.tar.gz"
    )


def source_dir_name(dolfinx_version: str = DOLFINX_VERSION) -> str:
    """Return the directory name the source tarball unpacks to.

    Args:
        dolfinx_version: The DOLFINx release being built.

    Returns:
        The directory name, such as ``dolfinx-0.11.0.post0``.
    """
    return f"dolfinx-{dolfinx_version}"


def cpp_source_dir(tarball_dir: Path) -> Path:
    """Return the C++ core's source directory inside an unpacked tarball."""
    return tarball_dir / "cpp"


def python_source_dir(tarball_dir: Path) -> Path:
    """Return the bindings' source directory inside an unpacked tarball."""
    return tarball_dir / "python"


def cpp_configure_arguments(
    *,
    source_dir: Path,
    build_dir: Path,
    prefix: Path,
    mpi_prefix: Path,
    python: str = sys.executable,
) -> list[str]:
    """Return the CMake configure command for the DOLFINx C++ core.

    Args:
        source_dir: The ``cpp`` directory of the unpacked source tree.
        build_dir: Out-of-tree build directory.
        prefix: Shared install prefix, holding everything to build against.
        mpi_prefix: Prefix holding the MPI compiler wrappers.
        python: Interpreter DOLFINx's CMake asks for its Basix and UFCx hints
            — the build venv's, where fenics-basix and fenics-ffcx are
            installed.

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
        "-DBUILD_SHARED_LIBS=ON",
        f"-DCMAKE_C_COMPILER={mpi_bin / 'mpicc'}",
        f"-DCMAKE_CXX_COMPILER={mpi_bin / 'mpicxx'}",
        f"-DCMAKE_PREFIX_PATH={prefix}",
        # Basix and UFCx come from the wheels in the build venv — they are the
        # upstream trio this wheel depends on and never re-vendors (spec §10),
        # so DOLFINx's Python-hint discovery is exactly the right path here.
        f"-DPython3_EXECUTABLE={python}",
        "-DDOLFINX_BASIX_PYTHON=ON",
        "-DDOLFINX_UFCX_PYTHON=ON",
        # The parallel HDF5 PETSc's configure built, in this same prefix.
        f"-DHDF5_ROOT={prefix}",
        # Every optional package, named. The five that are ON are the wheel's
        # feature set (spec §7); ParMETIS is the one exclusion, and it has to
        # be said, because DOLFINx's default for it is ON.
        "-DDOLFINX_ENABLE_PETSC=ON",
        "-DDOLFINX_ENABLE_SLEPC=ON",
        "-DDOLFINX_ENABLE_ADIOS2=ON",
        "-DDOLFINX_ENABLE_SCOTCH=ON",
        "-DDOLFINX_ENABLE_KAHIP=ON",
        "-DDOLFINX_ENABLE_SUPERLU_DIST=ON",
        "-DDOLFINX_ENABLE_PARMETIS=OFF",
        # DOLFINx's own unit tests and demos are not what this build ships.
        # The package-detection test programs are a different thing and stay
        # on: DOLFINX_SKIP_BUILD_TESTS would turn off the one in
        # FindKaHIP.cmake, which compiles and *runs* a partitioning call, and
        # that is the cheapest proof KaHIP is usable rather than merely
        # present.
        "-DBUILD_TESTING=OFF",
        "-DDOLFINX_SKIP_BUILD_TESTS=OFF",
    ]


def cpp_build_environment(
    *, prefix: Path, environ: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the environment the C++ build's package discovery needs.

    DOLFINx finds PETSc, SLEPc and SuperLU_DIST through ``pkg-config``, and
    builds the search path from ``PETSC_DIR`` and ``SLEPC_DIR`` rather than
    from a CMake variable. ``PETSC_ARCH`` is set to the empty string for the
    reason :mod:`wheelbuild.slepc` gives: a prefix install has no arch
    directory, and an inherited value would send the search inside one.

    Args:
        prefix: The shared install prefix.
        environ: Environment to extend. Defaults to this process's.

    Returns:
        A complete environment for the C++ build.
    """
    return {
        **(os.environ if environ is None else environ),
        "PETSC_DIR": str(prefix),
        "SLEPC_DIR": str(prefix),
        "PETSC_ARCH": "",
        "PKG_CONFIG_PATH": str(prefix / "lib" / "pkgconfig"),
    }


def cpp_build_arguments(build_dir: Path, jobs: int) -> list[str]:
    """Return the command that compiles the configured C++ core."""
    return ["cmake", "--build", str(build_dir), "--parallel", str(jobs)]


def cpp_install_arguments(build_dir: Path) -> list[str]:
    """Return the command that installs the built C++ core."""
    return ["cmake", "--install", str(build_dir)]


#: The search path ``libdolfinx`` is given, in search order. ``$ORIGIN`` is
#: where every library it links sits — PETSc, SLEPc, ADIOS2, KaHIP, PT-SCOTCH,
#: HDF5 — in the prefix now and in the wheel later. The second entry is
#: basix's, two levels up from ``dolfinx_solver/lib`` and back down, for the
#: reason :data:`BASIX_LIBRARY_DIR` gives.
LIBRARY_RPATH_ENTRIES = (
    "$ORIGIN",
    "$ORIGIN/../../" + BASIX_LIBRARY_DIR.as_posix(),
)


def library_rpath(observed: Sequence[str]) -> str:
    """Return the search path ``libdolfinx`` should carry, given what it has.

    :data:`LIBRARY_RPATH_ENTRIES` goes on the front and everything already
    there is kept. What is kept is CMake's own answer, which includes the
    absolute path of the basix wheel in the build venv: the relative entry is
    right for the installed wheel, and the absolute one is what resolves
    during the build, where ``libdolfinx`` sits in the prefix rather than in
    the layout the relative path describes.

    Args:
        observed: The library's current search path, in search order.

    Returns:
        The colon-separated path to set.
    """
    return ":".join(
        [
            *LIBRARY_RPATH_ENTRIES,
            *(entry for entry in observed if entry not in LIBRARY_RPATH_ENTRIES),
        ]
    )


def patchelf_arguments(library: Path, rpath: str) -> list[str]:
    """Return the command that rewrites a binary's rpath.

    Args:
        library: The built library or extension module.
        rpath: The search path to set.

    Returns:
        The argument vector.
    """
    return ["patchelf", "--set-rpath", rpath, str(library)]


def missing_artefacts(prefix: Path) -> list[Path]:
    """Return the required artefacts a C++ install prefix does not have.

    Args:
        prefix: DOLFINx install prefix.

    Returns:
        The missing paths, relative to the prefix, in declaration order.
    """
    return [
        relative for relative in REQUIRED_ARTEFACTS if not (prefix / relative).exists()
    ]


def pkgconfig_field(pkgconfig_text: str, field: str) -> str | None:
    """Return one field or variable of a pkg-config file.

    Args:
        pkgconfig_text: Contents of a ``.pc`` file.
        field: The name before the separator, such as ``Version`` or
            ``definitions``.

    Returns:
        The value with surrounding space removed, or ``None`` when the file
        does not carry that name.
    """
    for line in pkgconfig_text.splitlines():
        for separator in (":", "="):
            name, found, value = line.partition(separator)
            if found and name.strip() == field:
                return value.strip()
    return None


def pkgconfig_defines(pkgconfig_text: str) -> frozenset[str]:
    """Return the macro names a ``.pc`` file records the library as built with.

    DOLFINx writes its target's ``INTERFACE_COMPILE_DEFINITIONS`` into the
    ``definitions`` variable of its pkg-config file, one ``-D`` token each.
    That is where a feature that was enabled but not found shows up as
    absent — the installed file's own account of what the compile saw.

    Args:
        pkgconfig_text: Contents of the installed ``dolfinx.pc``.

    Returns:
        The macro names, with any ``=value`` dropped.
    """
    definitions = pkgconfig_field(pkgconfig_text, "definitions")
    if definitions is None:
        return frozenset()
    return frozenset(
        token.removeprefix("-D").partition("=")[0]
        for token in definitions.split()
        if token.startswith("-D")
    )


def required_sonames(prefix: Path) -> dict[str, str]:
    """Return the soname of each library ``libdolfinx`` has to be linked to.

    Args:
        prefix: The shared install prefix.

    Returns:
        Component name mapped to the soname read off the copy in this prefix.
        A library the prefix does not carry is left out: the missing-artefact
        check of the stage that owns it is the one that should report it.

    Raises:
        subprocess.CalledProcessError: When the ELF tools cannot read one.
    """
    sonames = {"MPI": mpich.MPI_SONAME}
    for component, relative in REQUIRED_LINKAGE.items():
        library = prefix / relative
        if not library.exists():
            continue
        found, _ = elf.read_dynamic(library)
        if found is not None:
            sonames[component] = found
    return sonames


def observe(prefix: Path) -> Build:
    """Read what a built DOLFINx C++ install says about itself.

    Args:
        prefix: DOLFINx install prefix.

    Returns:
        The facts :func:`build_problem` judges.

    Raises:
        FileNotFoundError: When ``dolfinx.pc`` is missing.
        subprocess.CalledProcessError: When the ELF tools cannot read
            ``libdolfinx``.
    """
    pkgconfig = (prefix / "lib" / "pkgconfig" / "dolfinx.pc").read_text(
        encoding="utf-8"
    )
    library = prefix / "lib" / "libdolfinx.so"
    _, needed = elf.read_dynamic(library)
    return Build(
        version=pkgconfig_field(pkgconfig, "Version"),
        defines=pkgconfig_defines(pkgconfig),
        runpath=elf.read_runpath(library),
        needed=needed,
    )


def build_problem(
    build: Build,
    *,
    sonames: dict[str, str],
    expected_version: str = RELEASE,
) -> str | None:
    """Report a DOLFINx that is not the one this wheel ships.

    Args:
        build: What the install says about itself.
        sonames: Component name mapped to the soname it must be linked to, as
            :func:`required_sonames` reads them out of the prefix.
        expected_version: The upstream release this wheel mirrors, without a
            packaging segment.

    Returns:
        A message naming the first thing that is wrong, or ``None``.
    """
    problems = (
        _release_problem(build, expected_version),
        _feature_problem(build),
        _rpath_problem(build),
        _linkage_problem(build, sonames),
    )
    return next((problem for problem in problems if problem is not None), None)


def _release_problem(build: Build, expected_version: str) -> str | None:
    """Report a prefix holding a DOLFINx other than the one this wheel ships."""
    if build.version is None:
        return (
            "the installed dolfinx.pc carries no Version, so nothing here can "
            "confirm which DOLFINx this prefix holds."
        )
    if build.version != expected_version:
        return (
            f"this prefix holds DOLFINx {build.version}, but this wheel ships "
            f"{expected_version} — the version it mirrors is the version it "
            "publishes (spec §3), so the two cannot differ."
        )
    return None


def _feature_problem(build: Build) -> str | None:
    """Report a package the build lost, or a licence that sinks the wheel."""
    for define, component in REQUIRED_DEFINES.items():
        if define not in build.defines:
            return (
                f"this DOLFINx was built without {component} (dolfinx.pc does "
                f"not record {define}). DOLFINx enables every optional "
                "package by default and downgrades a missing one to a log "
                "line, so a feature absent here is a feature the wheel has "
                "quietly lost — read the CMake output for what "
                f"{component} tripped over (spec §2)."
            )

    for define, reason in FORBIDDEN_DEFINES.items():
        if define in build.defines:
            return f"dolfinx.pc records {define}: {reason}."
    return None


def _rpath_problem(build: Build) -> str | None:
    """Report a core library that would not find its libraries in a wheel.

    Only the relative entries are checked. The absolute ones CMake baked in
    resolve inside the build container, which is exactly why their presence
    proves nothing about the wheel — and removing them is the assembly step's
    business, not this stage's.
    """
    missing = [entry for entry in LIBRARY_RPATH_ENTRIES if entry not in build.runpath]
    if not missing:
        return None
    found = ", ".join(build.runpath) or "nothing"
    return (
        f"libdolfinx looks for its libraries in {found}, which is missing "
        f"{', '.join(missing)}. Those entries are how it finds its siblings "
        "and basix once it is inside a wheel, where the build container's "
        "absolute paths beside them mean nothing."
    )


def _linkage_problem(build: Build, sonames: dict[str, str]) -> str | None:
    """Report a core library bound to a copy the wheel does not ship."""
    for component, soname in sonames.items():
        if soname not in build.needed:
            found = ", ".join(sorted(build.needed)) or "nothing"
            return (
                f"libdolfinx does not ask the loader for {soname} "
                f"({component}); it needs {found}. The wheel ships one copy "
                "of each of these libraries and libdolfinx has to be the one "
                "bound to them, not to another copy that was on the build "
                "machine."
            )
    return None


def validate_cpp(prefix: Path) -> Path:
    """Check a built DOLFINx C++ install against what the wheel needs.

    Args:
        prefix: DOLFINx install prefix.

    Returns:
        The validated prefix.

    Raises:
        FileNotFoundError: When the install is incomplete.
        ValueError: When the build is not the DOLFINx this wheel ships.
    """
    missing = missing_artefacts(prefix)
    if missing:
        raise FileNotFoundError(
            f"incomplete DOLFINx install at {prefix}: missing "
            + ", ".join(str(relative) for relative in missing)
        )

    problem = build_problem(observe(prefix), sonames=required_sonames(prefix))
    if problem is not None:
        raise ValueError(problem)
    return prefix


def wheel_arguments(
    *,
    source_dir: Path,
    build_dir: Path,
    prefix: Path,
    wheelhouse: Path,
    python: str = sys.executable,
) -> list[str]:
    """Return the command that builds the bindings wheel.

    ``wheel.py-api=cp312`` is the whole limited-API mechanism (spec §4); the
    rest points scikit-build-core's CMake run at the prefix everything else
    was installed into. Nothing is downloaded: ``--no-build-isolation`` keeps
    the build on the scikit-build-core, nanobind and mpi4py the container
    installed, and ``--no-deps`` keeps pip from resolving DOLFINx's own
    runtime dependencies — including the PyPI ``petsc4py`` this wheel must
    never carry.

    Args:
        source_dir: The ``python`` directory of the unpacked source tree.
        build_dir: Directory scikit-build-core builds in, kept across runs so
            a rebuild is incremental.
        prefix: Shared install prefix, holding the C++ core.
        wheelhouse: Where to leave the built wheel.
        python: Interpreter to run pip from — the build venv's.

    Returns:
        The argument vector.
    """
    mpi_bin = prefix / "bin"
    return [
        python,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(wheelhouse),
        "-C",
        f"wheel.py-api={bindings.LIMITED_API_TAG}",
        "-C",
        "cmake.build-type=Release",
        "-C",
        f"build-dir={build_dir}",
        "-C",
        f"cmake.define.CMAKE_PREFIX_PATH={prefix}",
        "-C",
        f"cmake.define.CMAKE_C_COMPILER={mpi_bin / 'mpicc'}",
        "-C",
        f"cmake.define.CMAKE_CXX_COMPILER={mpi_bin / 'mpicxx'}",
        str(source_dir),
    ]


def basix_library_dir(python: str = sys.executable) -> Path:
    """Return where the installed ``fenics-basix`` wheel keeps ``libbasix``.

    Args:
        python: Interpreter whose environment the basix wheel is installed in
            — the build venv's.

    Returns:
        The directory, such as
        ``<site-packages>/basix/lib``.

    Raises:
        subprocess.CalledProcessError: When basix is not importable there.
    """
    package = capture(
        [python, "-c", "import basix, sys; sys.stdout.write(basix.__file__)"]
    )
    return Path(package).parent / BASIX_LIBRARY_DIR.name


def prepare_site(prefix: Path, *, python: str = sys.executable) -> Path:
    """Lay the staging site out the way an installed environment is laid out.

    The bindings stage already links the vendored library directory into the
    site. DOLFINx needs one more stand-in: ``libbasix`` comes from the
    ``fenics-basix`` wheel, which installs as a sibling of our packages in
    site-packages, so :data:`RELATIVE_RPATH` reaches it through a ``basix``
    directory beside ``dolfinx``. Linking the build venv's basix into the site
    is what makes the check exercise that relative path instead of whatever
    ``LD_LIBRARY_PATH`` held.

    Args:
        prefix: The shared install prefix.
        python: Interpreter the basix wheel is installed for.

    Returns:
        The staging site, ready for the package to be unpacked into.

    Raises:
        NotADirectoryError: When the link's place is a real directory, which
            would mean something staged a basix of its own there.
    """
    site = bindings.prepare_site(prefix)
    link = site / BASIX_LIBRARY_DIR
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise NotADirectoryError(
            f"{link} is a real directory, not the link to the fenics-basix "
            "wheel this stage makes. The upstream trio is depended on and "
            "never re-vendored (spec §10), so a copy of basix inside the "
            "payload is not something this build produces."
        )
    # Relative, as the bindings stage's link is: the staging site and the
    # build venv both live under the build root, and a cache directory that
    # is mounted somewhere else next time would leave an absolute link
    # dangling.
    link.symlink_to(
        os.path.relpath(basix_library_dir(python), link.parent),
        target_is_directory=True,
    )
    return site


def built_wheel(wheelhouse: Path) -> Path:
    """Return the bindings wheel the build just left in the wheelhouse.

    Args:
        wheelhouse: Directory the build writes wheels into.

    Returns:
        The newest DOLFINx wheel there.

    Raises:
        FileNotFoundError: When the build left none.
    """
    wheels = sorted(
        wheelhouse.glob(f"{DISTRIBUTION}-*.whl"), key=lambda path: path.stat().st_mtime
    )
    if not wheels:
        raise FileNotFoundError(
            f"no {DISTRIBUTION} wheel in {wheelhouse}: the build reported "
            "success and produced nothing."
        )
    return wheels[-1]


def staged_paths(dolfinx_version: str = DOLFINX_VERSION) -> tuple[Path, Path]:
    """Return the two directories a staged DOLFINx occupies.

    Args:
        dolfinx_version: The release being staged.

    Returns:
        The package directory and its ``.dist-info``, relative to the staging
        site.
    """
    return Path(IMPORT_NAME), Path(f"{DISTRIBUTION}-{dolfinx_version}.dist-info")


def required_staged_artefacts(
    dolfinx_version: str = DOLFINX_VERSION,
) -> tuple[Path, ...]:
    """Return what a finished staging tree has to contain.

    The vendored library is on the list because the rpath is relative: the
    extension reaches ``libdolfinx`` through the staging tree, so an extension
    without it beside is an import that fails at load time. The ``METADATA``
    is on it because ``dolfinx/__init__.py`` reads its own version out of
    installed distribution metadata and raises without it.

    Args:
        dolfinx_version: The release being staged.

    Returns:
        The paths, relative to the staging site, in declaration order.
    """
    package, dist_info = staged_paths(dolfinx_version)
    return (
        package / "__init__.py",
        EXTENSION_PATH,
        dist_info / "METADATA",
        bindings.VENDORED_LIBRARY_DIR / soname(),
    )


def unpack(wheel: Path, site: Path, dolfinx_version: str = DOLFINX_VERSION) -> Path:
    """Unpack the built bindings wheel into the staging site.

    Everything is kept except the ``RECORD``; see the module docstring for why
    the ``.dist-info`` stays and why its file list does not.

    Args:
        wheel: The built wheel.
        site: The staging site.
        dolfinx_version: The release being staged.

    Returns:
        The staged package directory.
    """
    package, dist_info = staged_paths(dolfinx_version)
    for relative in (package, dist_info):
        staged = site / relative
        if staged.exists():
            shutil.rmtree(staged)

    print(f"+ unpack {wheel} into {site}", flush=True)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if Path(member).name == "RECORD":
                continue
            archive.extract(member, site)

    # Zip files carry no permissions the extractor keeps, and the wheel's own
    # libraries are executable.
    for library in (site / package).rglob("*.so"):
        library.chmod(0o755)
    return site / package


def missing_staged_artefacts(
    site: Path, dolfinx_version: str = DOLFINX_VERSION
) -> list[Path]:
    """Return the required staged artefacts a site does not have.

    Args:
        site: The staging site.
        dolfinx_version: The release being staged.

    Returns:
        The missing paths, relative to the site, in declaration order.
    """
    return [
        relative
        for relative in required_staged_artefacts(dolfinx_version)
        if not (site / relative).exists()
    ]


def staged_basix_extension(site: Path) -> Path:
    """Return the basix extension the staging site's basix link points at.

    Args:
        site: The staging site, whose ``basix/lib`` links to the installed
            ``fenics-basix`` wheel's library directory.

    Returns:
        The path of that wheel's own nanobind extension.
    """
    return (site / BASIX_LIBRARY_DIR).resolve().parent / BASIX_EXTENSION.name


def observe_staged(site: Path) -> Staged:
    """Read what the staged extension says about itself.

    Args:
        site: The staging site.

    Returns:
        The facts :func:`staged_problem` judges.

    Raises:
        subprocess.CalledProcessError: When ``readelf`` cannot read the
            extension.
    """
    extension = site / EXTENSION_PATH
    _, needed = elf.read_dynamic(extension)
    return Staged(runpath=elf.read_runpath(extension), needed=needed)


def staged_problem(staged: Staged, *, core_soname: str = soname()) -> str | None:
    """Report an extension that would not find its libraries in a wheel.

    Args:
        staged: What the staged extension says about itself.
        core_soname: The core library the extension has to ask for. Defaults
            to the soname this release declares.

    Returns:
        A message naming the first thing that is wrong, or ``None``.
    """
    if not set(RELATIVE_RPATH_ENTRIES).issubset(staged.runpath):
        found = ", ".join(staged.runpath) or "nothing"
        return (
            f"{EXTENSION_PATH} looks for its libraries in {found} rather than "
            f"{RELATIVE_RPATH}. CMake bakes in the absolute paths of the "
            "machine that built it, which exist on no user's machine; the "
            "rpath is rewritten to the wheel's own sibling layout after the "
            "build (spec §6)."
        )

    absolute = [entry for entry in staged.runpath if not entry.startswith("$ORIGIN")]
    if absolute:
        return (
            f"{EXTENSION_PATH} still carries the build path(s) "
            f"{', '.join(absolute)} on its rpath. A path that exists only in "
            "the build container either resolves to nothing on a user's "
            "machine or, worse, to somebody else's DOLFINx."
        )

    if core_soname not in staged.needed:
        found = ", ".join(sorted(staged.needed)) or "nothing"
        return (
            f"{EXTENSION_PATH} does not ask the loader for {core_soname} (it "
            f"needs {found}). The bindings are built against the C++ core "
            "this build installed, and an extension that names another one "
            "is a build that found something else on the machine."
        )
    return None


def validate_staged(site: Path, dolfinx_version: str = DOLFINX_VERSION) -> Path:
    """Check the staged DOLFINx against everything the wheel needs to be true.

    Args:
        site: The staging site.
        dolfinx_version: The release being staged.

    Returns:
        The validated staging site.

    Raises:
        FileNotFoundError: When the staging tree is incomplete.
        ValueError: When the extension would not find its libraries in a
            wheel.
    """
    missing = missing_staged_artefacts(site, dolfinx_version)
    if missing:
        raise FileNotFoundError(
            f"incomplete DOLFINx staging tree at {site}: missing "
            + ", ".join(str(relative) for relative in missing)
        )

    problem = staged_problem(observe_staged(site))
    if problem is None:
        problem = nanobind_problem(
            nanobind_abi_tag(site / EXTENSION_PATH),
            nanobind_abi_tag(staged_basix_extension(site)),
        )
    if problem is not None:
        raise ValueError(problem)
    return site


def build_cpp(
    *,
    tarball_dir: Path,
    build_dir: Path,
    prefix: Path,
    jobs: int,
    python: str = sys.executable,
) -> Path:
    """Configure, build, install and validate the DOLFINx C++ core.

    Args:
        tarball_dir: Unpacked DOLFINx source tree.
        build_dir: Out-of-tree build directory.
        prefix: Shared install prefix.
        jobs: Parallel build jobs.
        python: Interpreter DOLFINx's CMake takes its Basix and UFCx hints
            from.

    Returns:
        The validated install prefix.
    """
    environment = cpp_build_environment(prefix=prefix)
    check_call(
        cpp_configure_arguments(
            source_dir=cpp_source_dir(tarball_dir),
            build_dir=build_dir,
            prefix=prefix,
            mpi_prefix=prefix,
            python=python,
        ),
        env=environment,
    )
    check_call(cpp_build_arguments(build_dir, jobs), env=environment)
    check_call(cpp_install_arguments(build_dir), env=environment)

    library = prefix / "lib" / "libdolfinx.so"
    check_call(patchelf_arguments(library, library_rpath(elf.read_runpath(library))))
    return validate_cpp(prefix)


def build_bindings(
    *,
    tarball_dir: Path,
    build_dir: Path,
    prefix: Path,
    site: Path,
    wheelhouse: Path,
    python: str = sys.executable,
) -> Path:
    """Build, stage, re-rpath and validate the nanobind bindings.

    Args:
        tarball_dir: Unpacked DOLFINx source tree.
        build_dir: Directory scikit-build-core builds in.
        prefix: Shared install prefix, holding the C++ core.
        site: The staging site.
        wheelhouse: Where built wheels are left.
        python: Interpreter to run pip from — the build venv's.

    Returns:
        The validated staging site.

    Raises:
        ValueError: When the build is not limited-API, or when the staged
            extension would not find its libraries in a wheel.
    """
    check_call(
        wheel_arguments(
            source_dir=python_source_dir(tarball_dir),
            build_dir=build_dir,
            prefix=prefix,
            wheelhouse=wheelhouse,
            python=python,
        ),
        env=cpp_build_environment(prefix=prefix),
    )
    wheel = built_wheel(wheelhouse)

    problem = bindings.wheel_tag_problem(wheel.name)
    if problem is not None:
        raise ValueError(problem)

    unpack(wheel, site)
    check_call(patchelf_arguments(site / EXTENSION_PATH, RELATIVE_RPATH))
    return validate_staged(site)


def import_check_arguments(*, site: Path, python: str = sys.executable) -> list[str]:
    """Return the command that imports the whole staged stack in one process.

    It is the bindings' own check with DOLFINx added: the ordering being
    proven is the same one (spec §5), and this stage is the first at which
    ``dolfinx`` is there to import.

    Args:
        site: The staging site the packages are imported from.
        python: Interpreter to run — the build venv's.

    Returns:
        The argument vector.
    """
    return [*bindings.import_check_arguments(site=site, python=python), "--dolfinx"]


def check_imports(site: Path, *, python: str = sys.executable) -> Path:
    """Import the staged stack in a fresh interpreter and check the result.

    Args:
        site: The staging site.
        python: Interpreter to run the check with.

    Returns:
        The staging site.

    Raises:
        subprocess.CalledProcessError: When the check fails.
    """
    check_call(
        import_check_arguments(site=site, python=python),
        env=bindings.import_check_environment(site=site),
    )
    return site


def run(
    *,
    tarball_dir: Path,
    cpp_build_dir: Path,
    python_build_dir: Path,
    prefix: Path,
    jobs: int,
    python: str = sys.executable,
) -> Path:
    """Build both halves of DOLFINx and prove the result imports.

    Args:
        tarball_dir: Unpacked DOLFINx source tree.
        cpp_build_dir: Build directory for the C++ core.
        python_build_dir: Build directory for the bindings.
        prefix: Shared install prefix.
        jobs: Parallel build jobs.
        python: Interpreter to build and check with — the build venv's.

    Returns:
        The staging site holding the finished package.
    """
    build_cpp(
        tarball_dir=tarball_dir,
        build_dir=cpp_build_dir,
        prefix=prefix,
        jobs=jobs,
        python=python,
    )
    site = prepare_site(prefix, python=python)
    wheelhouse = prefix / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    build_bindings(
        tarball_dir=tarball_dir,
        build_dir=python_build_dir,
        prefix=prefix,
        site=site,
        wheelhouse=wheelhouse,
        python=python,
    )
    return check_imports(site, python=python)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the DOLFINx build and its validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check an already-built prefix instead of building one",
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--cpp-build-dir", type=Path)
    parser.add_argument("--python-build-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.validate_only and (
        args.source_dir is None
        or args.cpp_build_dir is None
        or args.python_build_dir is None
    ):
        parser.error(
            "--source-dir, --cpp-build-dir and --python-build-dir are required to build"
        )

    try:
        if args.validate_only:
            validate_cpp(args.prefix)
            site = validate_staged(bindings.site_dir(args.prefix))
            check_imports(site)
        else:
            site = run(
                tarball_dir=args.source_dir,
                cpp_build_dir=args.cpp_build_dir,
                python_build_dir=args.python_build_dir,
                prefix=args.prefix,
                jobs=args.jobs,
            )
    except (FileNotFoundError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"DOLFINx {DOLFINX_VERSION} staged in {site}, "
        f"{bindings.LIMITED_API_TAG}-abi3, with "
        f"{', '.join(REQUIRED_DEFINES.values())}, against PETSc "
        f"{petsc.PETSC_VERSION}, SLEPc {slepc.SLEPC_VERSION}, ADIOS2 "
        f"{adios2.ADIOS2_VERSION} and KaHIP {kahip.KAHIP_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
