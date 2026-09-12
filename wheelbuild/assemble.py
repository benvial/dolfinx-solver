"""Turn the install prefix into the wheel that gets published.

Every stage before this one produced a piece of the stack in a shared install
prefix. This one decides what of it a user receives, and the decisions are the
interesting part — a wheel is not a copy of a build tree:

* **The payload is the DT_NEEDED closure, not the prefix.** ``$prefix/lib``
  holds 129 shared libraries; the wheel carries the 28 the three extension
  modules actually reach, computed by walking what each binary asks the loader
  for. ADIOS2 installs bundled libraries nothing links (enet, EVPath), PETSc's
  configure leaves a second SuperLU_DIST library behind, and the MPICH stage
  leaves the ``libmpi`` it built and the wheel discards. Copying the directory
  would ship all of it, with a licence obligation for each (spec §7).
* **libmpi and libbasix are excluded from the graft, deliberately.**
  ``auditwheel repair --exclude`` leaves their ``DT_NEEDED`` entries alone and
  copies nothing: ``libmpi.so.12`` comes from the PyPI ``mpich`` wheel that
  mpi4py loads first (spec §5, ADR-0002), and ``libbasix.so`` from the
  ``fenics-basix`` wheel this one depends on and never re-vendors (spec §10).
  A grafted copy of either would put a second one in the process.
* **The vendored MPICH halves are pruned of dependencies they never call.**
  MPICH's ``mpifort``/``mpicxx`` wrappers put the whole ch4 device link line
  on every library they build, so ``libmpifort`` and ``libmpicxx`` name UCX,
  libpciaccess and libatomic in their ``DT_NEEDED`` while referencing not one
  symbol from any of them. Left there, those entries would pull four UCX
  libraries into a wheel whose licensing table does not mention them, and make
  the wheel's Fortran half depend on transports it never touches. They are
  removed against symbol evidence — see :func:`prune_problem` and ADR-0003.
* **The rpaths the earlier stages wrote are the linkage design.** The sibling
  layout (spec §6) and the relative reach into the basix wheel are decisions
  made in tickets 04 and 05, and ``auditwheel repair`` rewrites rpaths for a
  living. It keeps an existing entry only when that entry resolves inside the
  wheel, which is exactly the set we want kept, so the repaired wheel is
  re-read afterwards and the entries are asserted rather than assumed.
* **There is exactly one ``.dist-info``, and DOLFINx's version is a literal.**
  ``dolfinx/__init__.py`` ends with ``version("fenics-dolfinx")``, so the
  bindings stage staged that distribution's metadata beside the package to
  keep the import working (ticket 05). A wheel cannot carry it: ``auditwheel``
  asserts it finds one ``.dist-info`` and *pip refuses to install* a wheel
  with two ("invalid wheel, multiple .dist-info directories found"). So the
  metadata is left out and the one line that needed it is rewritten to the
  release this wheel ships — the same treatment petsc4py's and slepc4py's
  metadata already gets, and for the same reason: a distribution this project
  does not publish has no business being registered by its wheel (ticket 14).
  It is a modification to an LGPL source file, and :data:`VERSION_SUBSTITUTION`
  is where it is written down.

What comes out is one file: ``dolfinx_solver_complex-<version>-cp312-abi3-
manylinux_2_34_x86_64.whl``, audited by ``auditwheel show`` and
``abi3audit --strict``, carrying a build-enforced ``THIRD-PARTY-NOTICES``
(:mod:`wheelbuild.notices`), and proven by installing it into a clean venv
beside the PyPI ``mpich`` wheel and importing the stack.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import zipfile
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from wheelbuild import bindings, dolfinx, elf, mpich, notices, petsc, version
from wheelbuild._process import capture, check_call

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

#: Repository root, which is where the source of our own package is copied
#: from and where the driver modules are imported from.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: What of the repository the base wheel is built from. The build runs against
#: a copy rather than the repository itself: setuptools writes ``build/`` and
#: an egg-info directory beside the sources, and the container runs as root
#: over a bind-mounted checkout, so building in place would leave root-owned
#: directories in someone's working tree. Naming the inputs also means no
#: stray file in the checkout can reach the payload.
SOURCE_INPUTS = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "LICENSE.GPL-3.0",
    "dolfinx_solver",
)

#: Payload packages grafted into the wheel as top-level siblings, under their
#: upstream import names (spec §1, §6).
PAYLOAD_PACKAGES = ("dolfinx", "petsc4py", "slepc4py")

#: Names that must never be copied out of the staging site. ``__pycache__``
#: would ship cp312 bytecode in a wheel that also serves 3.13 and 3.14;
#: ``basix`` and ``dolfinx_solver`` are the two symlinks the earlier stages
#: made so their import checks could prove a relative rpath, and neither is a
#: directory this wheel owns.
EXCLUDED_PAYLOAD_NAMES = frozenset({"__pycache__", "basix", "dolfinx_solver"})

#: Sonames ``auditwheel repair`` must neither copy nor rewrite. Both are
#: satisfied at run time by another wheel in the same environment: libmpi by
#: the PyPI ``mpich`` wheel mpi4py loads first, libbasix by ``fenics-basix``.
EXCLUDED_SONAMES = ("libmpi.so.*", "libbasix.so*")

#: The C++ binding half of MPICH, which arrives in the payload because
#: MPICH's ``mpicxx`` wrapper puts ``-lmpicxx`` on every C++ link and ADIOS2's
#: C++ libraries are therefore linked against it. Spelled here rather than in
#: :mod:`wheelbuild.mpich`, whose ``VENDORED_ARTEFACTS`` lists the Fortran
#: half alone: whether the MPI story is rewritten to say the wheel vendors
#: both is ticket 13's question, and this driver only refuses to graft it
#: silently.
CXX_SONAME = "libmpicxx.so.12"

#: The transport link line MPICH's wrappers add, as sonames.
_DEVICE_DEPENDENCIES = (
    "libucp.so.0",
    "libuct.so.0",
    "libucs.so.0",
    "libucm.so.0",
    "libpciaccess.so.0",
    "libatomic.so.1",
)

#: Dependencies the vendored MPICH libraries name but never call, and which
#: are therefore removed from their ``DT_NEEDED`` before the wheel is built.
#:
#: MPICH's compiler wrappers add the ch4 device's whole link line — UCX,
#: libpciaccess, libatomic — to every library in the build, including the
#: Fortran and C++ binding shims, which call nothing but ``libmpi``'s PMPI
#: entry points. :func:`prune_problem` checks that claim against the symbol
#: tables before anything is removed, so a future MPICH that really does put
#: transport code in these libraries fails the build instead of losing a
#: dependency.
OVER_LINKED = {
    mpich.FORTRAN_SONAME: _DEVICE_DEPENDENCIES,
    CXX_SONAME: _DEVICE_DEPENDENCIES,
}

#: The libraries a finished wheel is checked for, and why each is refused.
#:
#: Both are excluded from the graft, so neither should ever be here — but
#: ``--exclude`` is one flag and the build prefix holds a ``libmpi`` of its
#: own, so the wheel is read back rather than trusted. Each is matched both
#: under its plain soname and under the ``libfoo-<hash>.so.N`` name
#: ``auditwheel`` gives what it copies in, since a grafted copy is exactly the
#: failure being looked for. ``libmpifort`` and ``libmpicxx`` are deliberately
#: not matched: the patterns are anchored on the graft's hyphen for that
#: reason.
REFUSED_LIBRARIES = (
    (
        ("libmpi.so.*", "libmpi-*.so.*"),
        (
            "The MPI runtime comes from the PyPI mpich wheel that mpi4py "
            "loads first, and a second copy in the process is the failure "
            "the whole linkage design avoids (spec §5, ADR-0002). Either the "
            "--exclude was dropped or the build prefix's own libmpi reached "
            "the payload."
        ),
    ),
    (
        ("libbasix.so*", "libbasix-*.so*"),
        (
            "It belongs to the fenics-basix wheel, whose C++ objects cross "
            "the extension boundary into ours; two copies is two type "
            "registries (spec §10)."
        ),
    ),
)

#: Where a pruned dependency is looked for when its symbols are checked. The
#: prefix first, since UCX is built there by the MPICH stage, then the image's
#: own library directories for libpciaccess and libatomic.
SYSTEM_LIBRARY_DIRS = (Path("/usr/lib64"), Path("/lib64"), Path("/usr/lib"))

#: The two configuration files the bindings install carrying the absolute
#: build prefix, and the keys whose values are that path. They are inert —
#: ``petsc4py.lib`` reads only ``PETSC_ARCH`` out of them — but a published
#: wheel should not describe a directory that existed in a container for an
#: hour, and ``petsc4py.get_config()`` hands the whole file to whoever asks.
#: The vendored PETSc has no prefix a user can reach, so the honest value is
#: no value (ticket 04).
CONFIG_FILES = {
    Path("petsc4py/lib/petsc.cfg"): ("PETSC_DIR",),
    Path("slepc4py/lib/slepc.cfg"): ("PETSC_DIR", "SLEPC_DIR"),
}

#: The wheel's own distribution name, as a file name spells it.
DISTRIBUTION = "dolfinx_solver_complex"

#: Tag the assembled wheel carries before ``auditwheel`` decides which
#: manylinux it qualifies for: one interpreter tag, the stable ABI, and the
#: platform spelled the way a freshly-linked binary is (spec §4).
BUILD_TAG = f"{bindings.LIMITED_API_TAG}-abi3-linux_x86_64"

#: The platform tag the repaired wheel has to end up with (spec §2).
PLATFORM_TAG = "manylinux_2_34_x86_64"

#: Tag of the published wheel.
WHEEL_TAG = f"{bindings.LIMITED_API_TAG}-abi3-{PLATFORM_TAG}"

#: What the clean venv needs beyond the wheel itself, for the import check to
#: run: :mod:`wheelbuild.bindings` imports ``packaging``, and the check
#: imports the driver modules. Nothing here is a runtime dependency of the
#: wheel, which brings its own.
CHECK_REQUIREMENTS = ("packaging",)

#: The harvested notices, inside our own package so they install with it.
NOTICES_PATH = Path("dolfinx_solver") / "THIRD-PARTY-NOTICES"

#: Directory ``auditwheel`` grafts into, which is the distribution name plus
#: its default suffix. Named here because the validation reads what landed
#: there back out of the wheel.
GRAFT_DIR = f"{DISTRIBUTION}.libs"


class Wheel(NamedTuple):
    """What a built wheel says about itself.

    Attributes:
        names: Every member of the archive, in archive order.
        tag: The ``Tag`` its ``WHEEL`` metadata declares.
        libraries: File names of the shared libraries it carries, wherever
            they sit — the vendored directory or auditwheel's graft directory.
        runpaths: Search path of each ELF member, keyed by member name.
        needed: ``DT_NEEDED`` entries of each ELF member, keyed by member name.
        build_paths: Members whose text contains the build prefix. Binary
            members are not scanned: an unstripped library carries build paths
            in its debug information, which is not something a wheel promises
            to be free of, while a configuration file or a script that names
            the container's ``/build`` is a leak.
    """

    names: tuple[str, ...]
    tag: str | None
    libraries: tuple[str, ...]
    runpaths: Mapping[str, tuple[str, ...]]
    needed: Mapping[str, frozenset[str]]
    build_paths: tuple[str, ...]


def wheel_name(tag: str, distribution_version: str = version.__version__) -> str:
    """Return the file name of a wheel with one tag.

    Args:
        tag: The compatibility tag, such as ``cp312-abi3-linux_x86_64``.
        distribution_version: The version being packaged.

    Returns:
        The file name.
    """
    return f"{DISTRIBUTION}-{distribution_version}-{tag}.whl"


def dist_info_name(distribution_version: str = version.__version__) -> str:
    """Return the name of our own ``.dist-info`` directory.

    Args:
        distribution_version: The version being packaged.

    Returns:
        The directory name, as it appears inside the wheel.
    """
    return f"{DISTRIBUTION}-{distribution_version}.dist-info"


def source_copy(destination: Path, repo_root: Path = REPO_ROOT) -> Path:
    """Copy the base wheel's build inputs out of the repository.

    Args:
        destination: Directory to build the base wheel in. Replaced if it
            exists, so a warm cache cannot leave a stale copy of the package.
        repo_root: The repository to copy from.

    Returns:
        The directory the inputs were copied into.

    Raises:
        FileNotFoundError: When a declared input is missing, which means the
            repository layout and :data:`SOURCE_INPUTS` have drifted apart.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for name in SOURCE_INPUTS:
        source = repo_root / name
        if not source.exists():
            raise FileNotFoundError(
                f"{source} is named in SOURCE_INPUTS and is not in the "
                "repository, so the base wheel would be built from a "
                "different set of files than this driver describes."
            )
        if source.is_dir():
            shutil.copytree(
                source,
                destination / name,
                ignore=shutil.ignore_patterns("__pycache__", "lib"),
            )
        else:
            shutil.copy2(source, destination / name)
    return destination


def base_wheel_arguments(
    source_dir: Path, wheelhouse: Path, *, python: str = sys.executable
) -> list[str]:
    """Return the command that builds the pure-Python base wheel.

    ``--no-deps`` and ``--no-build-isolation`` for the same reasons the
    bindings stage gives: nothing is downloaded, and the build stays on the
    setuptools the container installed.

    Args:
        source_dir: The copied source tree.
        wheelhouse: Where to leave the wheel.
        python: Interpreter to run pip from.

    Returns:
        The argument vector.
    """
    return [
        python,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(wheelhouse),
        str(source_dir),
    ]


def excluded(soname: str) -> bool:
    """Report whether a soname is one the wheel never carries.

    Args:
        soname: A ``DT_NEEDED`` entry.

    Returns:
        ``True`` when :data:`EXCLUDED_SONAMES` matches it.
    """
    return any(fnmatch(soname, pattern) for pattern in EXCLUDED_SONAMES)


def pruned(soname: str, needed: Iterable[str]) -> frozenset[str]:
    """Return what a library needs once its over-linked entries are dropped.

    Args:
        soname: File name of the library the entries belong to.
        needed: Its ``DT_NEEDED`` entries.

    Returns:
        The entries that survive. Unchanged for every library except the two
        in :data:`OVER_LINKED`.
    """
    removable = set(OVER_LINKED.get(soname, ()))
    return frozenset(entry for entry in needed if entry not in removable)


def payload_extensions(site: Path) -> list[Path]:
    """Return the extension modules the payload's closure starts from.

    These are the binaries a user's process loads: the three compiled modules
    in the staged packages. Everything else in the wheel is there because one
    of them, directly or transitively, asks the loader for it.

    Args:
        site: The staging site holding the payload packages.

    Returns:
        The extension modules, sorted.

    Raises:
        FileNotFoundError: When a payload package has no extension, which
            means an earlier stage staged a tree without its binary.
    """
    found = []
    for package in PAYLOAD_PACKAGES:
        extensions = sorted((site / package).rglob("*.abi3.so"))
        if not extensions:
            raise FileNotFoundError(
                f"no extension module under {site / package}: the payload is "
                "the closure of what the extensions need, so a package "
                "without one would contribute no libraries at all."
            )
        found.extend(extensions)
    return found


def closure(entry_points: Sequence[Path], library_dir: Path) -> dict[str, Path]:
    """Walk what a set of binaries needs, through the vendored library tree.

    The walk resolves each ``DT_NEEDED`` entry by name inside ``library_dir``
    and follows it. An entry that is not there is external — the toolchain,
    libc, the two excluded wheels' libraries — and is auditwheel's business,
    not this function's. Over-linked entries are dropped before the walk
    (:func:`pruned`), so what they drag in is never reached in the first place.

    The walk carries each library's soname alongside the file it resolved to,
    because in the prefix the two differ: ``libmpifort.so.12`` is a symlink to
    ``libmpifort.so.12.6.1``, and :data:`OVER_LINKED` is keyed — as every
    ``DT_NEEDED`` entry is — on the soname. Following the resolved name would
    ask about a library nothing links by that spelling and prune nothing.

    Args:
        entry_points: Binaries to start from.
        library_dir: The prefix's library directory.

    Returns:
        Soname to the real file it resolves to, for every library the wheel
        has to carry.
    """
    found: dict[str, Path] = {}
    queue = [(path.name, path) for path in entry_points]
    while queue:
        soname, path = queue.pop()
        for entry in pruned(soname, elf.read_dynamic(path)[1]):
            if entry in found or excluded(entry):
                continue
            candidate = library_dir / entry
            if not candidate.exists():
                continue
            found[entry] = candidate.resolve()
            queue.append((entry, candidate.resolve()))
    return found


def find_library(soname: str, library_dir: Path) -> Path | None:
    """Return where a soname resolves, for the symbol check.

    Args:
        soname: The library to find.
        library_dir: The prefix's library directory, searched first.

    Returns:
        The file, or ``None`` when it is on none of the search paths.
    """
    for directory in (library_dir, *SYSTEM_LIBRARY_DIRS):
        candidate = directory / soname
        if candidate.exists():
            return candidate.resolve()
    return None


def prune_problem(
    library: Path, removable: Sequence[str], library_dir: Path
) -> str | None:
    """Report a dependency that cannot be pruned because it is really used.

    The claim :data:`OVER_LINKED` makes is that these libraries appear on a
    link line and nowhere in the code. The evidence is the symbol tables: if
    none of the library's undefined symbols is defined by the dependency, then
    removing the entry removes nothing the loader would have had to satisfy.

    Args:
        library: The library whose entries are to be pruned.
        removable: The sonames to be removed from it.
        library_dir: Where to look for those sonames.

    Returns:
        A message naming the first dependency that is actually referenced, or
        that could not be found to check, and ``None`` when every one of them
        is safe to drop.
    """
    _, undefined = elf.read_symbols(library)
    for soname in removable:
        found = find_library(soname, library_dir)
        if found is None:
            return (
                f"{soname} is listed as an over-linked dependency of "
                f"{library.name} and is not on any search path, so the claim "
                "that nothing references it cannot be checked. Pruning a "
                "DT_NEEDED entry on an unchecked assumption is how a wheel "
                "ends up failing to load in someone else's process."
            )
        defined, _ = elf.read_symbols(found)
        used = sorted(undefined & defined)
        if used:
            return (
                f"{library.name} references {len(used)} symbols from "
                f"{soname} ({', '.join(used[:5])}), so this build is not the "
                "over-linking OVER_LINKED describes. MPICH has started "
                "putting code that calls its transports into the binding "
                "libraries, and either the wheel now has to vendor that "
                "dependency and account for its licence (spec §7), or the "
                "MPICH configure line has to change."
            )
    return None


def patchelf_arguments(library: Path, removable: Sequence[str]) -> list[str]:
    """Return the command that drops dependencies from a library.

    Args:
        library: The library to patch.
        removable: The sonames to remove.

    Returns:
        The ``patchelf`` argument vector.
    """
    command = ["patchelf"]
    for soname in removable:
        command.extend(["--remove-needed", soname])
    command.append(str(library))
    return command


def prune(library_dir: Path, staged: Path) -> list[str]:
    """Drop the over-linked dependencies from the staged MPICH libraries.

    Args:
        library_dir: The prefix's library directory, for resolving the
            dependencies whose symbols are checked.
        staged: The wheel's vendored library directory, holding the copies
            that are patched. The originals in the prefix are left alone, so
            a warm cache stays usable by the stages that built them.

    Returns:
        The sonames that were removed, as ``library: dependency`` lines, for
        the build log.

    Raises:
        ValueError: When a dependency turns out to be referenced after all;
            see :func:`prune_problem`.
    """
    removed: list[str] = []
    for soname, removable in sorted(OVER_LINKED.items()):
        library = staged / soname
        if not library.exists():
            continue
        present = [
            entry for entry in removable if entry in elf.read_dynamic(library)[1]
        ]
        if not present:
            continue
        # Only what the library really names is checked: an MPICH built
        # against a system UCX, or one whose C++ shim gets a narrower link
        # line than its Fortran one, leaves some of these off — and asking
        # whether an absent dependency is referenced would fail the build for
        # a library that is already as clean as this wants it.
        problem = prune_problem(library, present, library_dir)
        if problem is not None:
            raise ValueError(problem)
        check_call(patchelf_arguments(library, present))
        removed.extend(f"{soname}: {entry}" for entry in present)
    return removed


def blank_values(text: str, keys: Sequence[str]) -> str:
    """Return a key-value configuration file with some values emptied.

    Args:
        text: Contents of the file.
        keys: Keys whose values are to be blanked.

    Returns:
        The file, with those keys kept and their values gone. A key the file
        does not have is not added: this rewrites what upstream wrote rather
        than inventing configuration.
    """
    lines = []
    for line in text.splitlines():
        key = line.partition("=")[0]
        if key.strip() in keys:
            # The key keeps the padding upstream wrote, so the rewritten file
            # reads like the one it replaces.
            lines.append(f"{key}=")
        else:
            lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def rewrite_configs(root: Path) -> list[Path]:
    """Empty the absolute build prefix out of the bindings' configuration.

    Args:
        root: The staged wheel tree.

    Returns:
        The files that were rewritten.

    Raises:
        FileNotFoundError: When one of them is not in the payload, which
            would mean the bindings stage stopped installing it and this
            rewrite is describing a file that no longer exists.
    """
    rewritten = []
    for relative, keys in CONFIG_FILES.items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(
                f"{relative} is not in the payload. It carries the absolute "
                "build prefix and is rewritten here; a payload without it is "
                "a bindings install this driver no longer describes."
            )
        path.write_text(
            blank_values(path.read_text(encoding="utf-8"), keys), encoding="utf-8"
        )
        rewritten.append(relative)
    return rewritten


def wheel_metadata(tag: str) -> str:
    """Return the ``WHEEL`` metadata for the assembled wheel.

    The base wheel setuptools built is pure Python and says so. Ours carries
    compiled extensions in a platform-specific layout, so both the tag and
    ``Root-Is-Purelib`` have to change — a purelib wheel would install the
    vendored libraries into a directory the extensions' rpaths do not point
    at, on any interpreter whose purelib and platlib differ.

    Args:
        tag: The compatibility tag to declare.

    Returns:
        The file's contents.
    """
    return (
        "Wheel-Version: 1.0\n"
        f"Generator: wheelbuild.assemble ({version.__version__})\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {tag}\n"
    )


def record_entry(path: Path, name: str) -> str:
    """Return one ``RECORD`` line for a file.

    Args:
        path: The file on disk.
        name: Its name inside the wheel.

    Returns:
        The line, with the hash and size the wheel format asks for.
    """
    digest = hashlib.sha256(path.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"{name},sha256={encoded},{path.stat().st_size}"


def write_record(root: Path, dist_info: str) -> Path:
    """Write the ``RECORD`` of a staged wheel tree.

    Args:
        root: The staged tree.
        dist_info: Name of the ``.dist-info`` directory the record belongs to.

    Returns:
        The file that was written.
    """
    record = root / dist_info / "RECORD"
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != record
    )
    lines = [record_entry(root / name, name) for name in names]
    # The RECORD does not record its own hash: it cannot, and the format says
    # to leave both fields empty.
    lines.append(f"{record.relative_to(root).as_posix()},,")
    record.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return record


def unpack(archive: Path, destination: Path) -> Path:
    """Extract a wheel, keeping the executable bits its members carry.

    ``zipfile`` drops permissions on extraction, and the libraries in a wheel
    are executable.

    Args:
        archive: The wheel to extract.
        destination: Directory to extract into.

    Returns:
        The directory.
    """
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            extracted = Path(zipped.extract(member, destination))
            mode = member.external_attr >> 16
            if mode:
                extracted.chmod(mode & 0o777)
    return destination


def pack(root: Path, archive: Path) -> Path:
    """Zip a staged wheel tree, in a fixed order and with its modes.

    Args:
        root: The staged tree.
        archive: The wheel to write.

    Returns:
        The wheel.
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    print(f"+ pack {len(names)} files into {archive}", flush=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name in names:
            info = zipfile.ZipInfo.from_file(root / name, name)
            info.compress_type = zipfile.ZIP_DEFLATED
            zipped.writestr(info, (root / name).read_bytes())
    return archive


def copy_payload(site: Path, root: Path) -> list[str]:
    """Copy the payload packages into the staged wheel tree.

    Args:
        site: The staging site the earlier stages filled.
        root: The staged wheel tree.

    Returns:
        The packages copied.
    """
    copied = []
    for package in PAYLOAD_PACKAGES:
        shutil.copytree(
            site / package,
            root / package,
            ignore=shutil.ignore_patterns(*EXCLUDED_PAYLOAD_NAMES),
            symlinks=False,
        )
        copied.append(package)
    return copied


def copy_libraries(libraries: Mapping[str, Path], root: Path) -> list[str]:
    """Copy the vendored libraries in under the sonames that reach them.

    Each library is copied to its soname rather than to the versioned file
    name it has in the prefix: the soname is what every ``DT_NEEDED`` asks
    for, and a wheel cannot carry the symlink chain that would otherwise
    bridge the two (spec §6's layout is resolved by name, and a zip has no
    symlinks worth relying on).

    Args:
        libraries: Soname to the real file, from :func:`closure`.
        root: The staged wheel tree.

    Returns:
        The sonames copied, sorted.
    """
    library_dir = root / bindings.VENDORED_LIBRARY_DIR
    library_dir.mkdir(parents=True, exist_ok=True)
    for soname, path in sorted(libraries.items()):
        destination = library_dir / soname
        shutil.copy2(path, destination)
        destination.chmod(0o755)
    return sorted(libraries)


#: The line of upstream's ``dolfinx/__init__.py`` that reads the package's
#: version out of installed distribution metadata, and what it becomes in the
#: payload. See the module docstring for why the metadata cannot ship; the
#: version this substitutes in is the same string that metadata carried, from
#: the one place this repository keeps it (spec §3).
VERSION_SUBSTITUTION = (
    'from importlib.metadata import version\n\n__version__ = version("fenics-dolfinx")',
    (
        "# Substituted by wheelbuild.assemble: the fenics-dolfinx distribution\n"
        "# metadata this line read is not in this wheel, because a wheel may\n"
        "# carry only one .dist-info and it is not ours to register.\n"
        f'__version__ = "{version.DOLFINX_VERSION}"'
    ),
)


def substitute_version(text: str) -> str:
    """Return upstream's ``dolfinx/__init__.py`` with its version made a literal.

    Args:
        text: Contents of the staged ``dolfinx/__init__.py``.

    Returns:
        The file with :data:`VERSION_SUBSTITUTION` applied.

    Raises:
        ValueError: When the line is not there to substitute. That means
            upstream has changed how the package learns its own version, and
            the payload would either import a version from nowhere or raise
            ``PackageNotFoundError`` at the user's first import — so it is a
            build failure rather than a substitution to skip.
    """
    original, replacement = VERSION_SUBSTITUTION
    if original not in text:
        raise ValueError(
            "the staged dolfinx/__init__.py does not contain the "
            f"{original.splitlines()[-1]!r} this assembly rewrites. DOLFINx "
            "reads its own version out of installed distribution metadata "
            "that this wheel cannot carry, so if upstream has changed how it "
            "does that, the new way has to be handled here rather than "
            "shipped untouched."
        )
    return text.replace(original, replacement)


#: How upstream's ``dolfinx/fem/petsc.py`` finds the PETSc shared library, and
#: what that lookup becomes in the payload. Upstream asks petsc4py for its
#: ``PETSC_DIR`` and joins ``lib/libpetsc.so`` onto it; this wheel blanks that
#: value out of ``petsc.cfg`` because it holds the container's build prefix
#: (:data:`CONFIG_FILES`), and there is no absolute path that would be right
#: instead — the library ships inside the wheel, wherever pip puts it. So the
#: candidate list becomes the one path the wheel has, resolved from the
#: module's own location: ``dolfinx/fem/petsc.py`` is three directories below
#: the wheel root, and the vendored library directory sits there beside the
#: payload packages (ticket 16).
#:
#: Upstream's own "could not find" ``RuntimeError`` is left standing below
#: this, so a wheel that somehow shipped without libpetsc still says so.
PETSC_LIB_SUBSTITUTION = (
    """    import petsc4py as _petsc4py

    petsc_dir = _petsc4py.get_config()["PETSC_DIR"]
    petsc_arch = _petsc4py.lib.getPathArchPETSc()[1]
    candidate_paths = [
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"),
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.dylib"),
    ]""",
    (
        "    # Substituted by wheelbuild.assemble: this wheel's PETSc is "
        "vendored\n"
        "    # inside it rather than installed under a PETSC_DIR, so the "
        "petsc.cfg\n"
        "    # upstream reads this from carries no prefix a user could reach.\n"
        "    candidate_paths = [\n"
        "        str(\n"
        "            pathlib.Path(__file__).resolve().parents[2]\n"
        f'            / "{bindings.VENDORED_LIBRARY_DIR.as_posix()}"\n'
        f'            / "{petsc.soname()}"\n'
        "        )\n"
        "    ]"
    ),
)


def substitute_petsc_lib(text: str) -> str:
    """Return upstream's ``dolfinx/fem/petsc.py`` aimed at the vendored PETSc.

    Args:
        text: Contents of the staged ``dolfinx/fem/petsc.py``.

    Returns:
        The file with :data:`PETSC_LIB_SUBSTITUTION` applied.

    Raises:
        ValueError: When the lookup is not there to substitute. The class body
            that calls it runs at import, so an unsubstituted wheel does not
            fail at the call that needs the library — it fails at ``import
            dolfinx.fem.petsc``, which is every solve, every assembly and
            every boundary condition in the package. Shipping that is worse
            than failing the build.
    """
    original, replacement = PETSC_LIB_SUBSTITUTION
    if original not in text:
        raise ValueError(
            "the staged dolfinx/fem/petsc.py does not look up libpetsc the "
            "way this assembly rewrites. DOLFINx resolves the PETSc shared "
            "library through petsc4py's PETSC_DIR, which this wheel cannot "
            "fill in, so the lookup is replaced with the vendored library's "
            "own path; upstream having changed it means the new way has to "
            "be handled here rather than shipped untouched."
        )
    return text.replace(original, replacement)


def library_rpath(soname: str) -> str:
    """Return the search path a vendored library gets inside the wheel.

    Every library in the payload sits in one directory, so ``$ORIGIN`` is the
    whole answer for all but one of them: ``libdolfinx`` also reaches into the
    ``fenics-basix`` wheel's own library directory, which is two levels up and
    back down (spec §10, ticket 05).

    Args:
        soname: File name of the staged library.

    Returns:
        The colon-separated search path to set.
    """
    if soname == dolfinx.soname():
        return ":".join(dolfinx.LIBRARY_RPATH_ENTRIES)
    return "$ORIGIN"


def rpath_arguments(library: Path, rpath: str) -> list[str]:
    """Return the command that rewrites a staged library's search path.

    Args:
        library: The staged library.
        rpath: The search path to set.

    Returns:
        The ``patchelf`` argument vector.
    """
    return ["patchelf", "--set-rpath", rpath, str(library)]


def set_library_rpaths(library_dir: Path) -> list[str]:
    """Give every vendored library a search path relative to the wheel.

    This is not tidying up after the compilers, it is what stops the wheel
    carrying two of everything. The libraries PETSc's configure builds — and
    the ones our own CMake stages install — record the absolute install prefix
    as their ``RUNPATH``: ``libpetsc``'s is ``/build/install/lib`` and nothing
    else. ``auditwheel repair`` resolves each ``DT_NEEDED`` entry the way the
    loader would, so with that path on the library it finds every dependency
    in the *build tree* rather than in the copy sitting beside it, calls it
    external and grafts a second copy under a hashed name. A first run of this
    stage produced a wheel with two of libpetsc, libhdf5, libscotch and twenty
    more.

    Rewriting the path to ``$ORIGIN`` also removes the build prefix from the
    published binaries, which is a thing this wheel should not carry anyway.

    Args:
        library_dir: The wheel's vendored library directory.

    Returns:
        The libraries whose path was rewritten, sorted.

    Raises:
        ValueError: When a library still carries an absolute entry
            afterwards, which would mean patchelf did not do what was asked.
    """
    rewritten = []
    for library in sorted(library_dir.iterdir()):
        if not library.is_file():
            continue
        check_call(rpath_arguments(library, library_rpath(library.name)))
        absolute = [
            entry for entry in elf.read_runpath(library) if entry.startswith("/")
        ]
        if absolute:
            raise ValueError(
                f"{library.name} still carries the absolute search path "
                f"{', '.join(absolute)} after being patched. Everything the "
                "wheel vendors has to be found relative to where it is "
                "installed, and an absolute build path is both a leak and the "
                "reason auditwheel would graft a second copy of every library "
                "this one links."
            )
        rewritten.append(library.name)
    return rewritten


def stage(
    *,
    prefix: Path,
    base_wheel: Path,
    staging: Path,
    build_root: Path,
    image_licenses: Path,
) -> Path:
    """Lay out everything the wheel contains, ready to be zipped.

    Args:
        prefix: The shared install prefix.
        base_wheel: The pure-Python wheel setuptools built from our package.
        staging: Directory to stage in. Replaced if it exists.
        build_root: Where the source trees the notices are harvested from are.
        image_licenses: The build image's licence directory.

    Returns:
        The staged tree.

    Raises:
        FileNotFoundError: When a payload package, a configuration file or a
            licence text is missing.
        ValueError: When a dependency cannot be pruned; see
            :func:`prune_problem`.
    """
    if staging.exists():
        shutil.rmtree(staging)
    unpack(base_wheel, staging)

    site = bindings.site_dir(prefix)
    copy_payload(site, staging)
    libraries = closure(payload_extensions(site), prefix / "lib")
    copy_libraries(libraries, staging)
    set_library_rpaths(staging / bindings.VENDORED_LIBRARY_DIR)
    prune(prefix / "lib", staging / bindings.VENDORED_LIBRARY_DIR)
    rewrite_configs(staging)

    package_init = staging / dolfinx.IMPORT_NAME / "__init__.py"
    package_init.write_text(
        substitute_version(package_init.read_text(encoding="utf-8")), encoding="utf-8"
    )
    petsc_layer = staging / dolfinx.IMPORT_NAME / "fem" / "petsc.py"
    petsc_layer.write_text(
        substitute_petsc_lib(petsc_layer.read_text(encoding="utf-8")), encoding="utf-8"
    )

    (staging / NOTICES_PATH).write_text(
        notices.harvest(notices.roots(build_root, image_licenses=image_licenses)),
        encoding="utf-8",
    )

    dist_info = dist_info_name()
    (staging / dist_info / "WHEEL").write_text(
        wheel_metadata(BUILD_TAG), encoding="utf-8"
    )
    top_level = staging / dist_info / "top_level.txt"
    if top_level.exists():
        top_level.write_text(
            "".join(
                f"{name}\n" for name in sorted([*PAYLOAD_PACKAGES, "dolfinx_solver"])
            ),
            encoding="utf-8",
        )
    write_record(staging, dist_info)
    return staging


def audit_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment ``auditwheel`` has to be run in.

    ``LD_LIBRARY_PATH`` is removed, and that is not tidiness — it decides what
    the wheel contains. auditwheel resolves each ``DT_NEEDED`` entry the way
    the loader does, and the loader searches ``DT_RPATH``, then
    ``LD_LIBRARY_PATH``, then ``DT_RUNPATH``. Every binary in this stack
    carries a ``RUNPATH`` (patchelf writes one, and so does the linker by
    default), so the build's ``LD_LIBRARY_PATH=$prefix/lib`` — exported by
    ``build-wheel.sh`` so each stage can link against the tree as it grows —
    is searched *before* the relative entries that point inside the wheel.
    Every vendored library then resolves to the prefix copy, is classified as
    external, and is grafted in a second time under a hashed name: a first run
    of this stage produced a wheel carrying two of libpetsc, libdolfinx,
    libhdf5 and twenty-odd more.

    Args:
        environment: The environment to start from. Defaults to this process's.

    Returns:
        A copy with ``LD_LIBRARY_PATH`` removed.
    """
    return {
        key: value
        for key, value in (os.environ if environment is None else environment).items()
        if key != "LD_LIBRARY_PATH"
    }


def repair_arguments(wheel: Path, wheelhouse: Path) -> list[str]:
    """Return the ``auditwheel repair`` command.

    Args:
        wheel: The assembled wheel.
        wheelhouse: Where the repaired wheel goes.

    Returns:
        The argument vector.
    """
    # --only-plat: the output carries exactly the platform tag spec §2 names.
    # Without it auditwheel adds every more-compatible tag the stack happens
    # to qualify for to the file name, and the finished wheel is then not the
    # file this driver goes looking for.
    command = ["auditwheel", "repair", "--plat", PLATFORM_TAG, "--only-plat"]
    for pattern in EXCLUDED_SONAMES:
        command.extend(["--exclude", pattern])
    command.extend(["--wheel-dir", str(wheelhouse), str(wheel)])
    return command


def show_arguments(wheel: Path) -> list[str]:
    """Return the ``auditwheel show`` command, reporting as JSON.

    The prose ``show`` prints cannot be checked by a build, and its headline
    is misleading here in a way worth being precise about: ``show`` has no
    ``--exclude``, so it counts ``libmpi.so.12`` and ``libbasix.so`` as
    unresolved external references and concludes the wheel is only consistent
    with ``linux_x86_64``. Those two are resolved at run time by the other
    wheels in the environment, on purpose (spec §5, §10). What the report can
    be asked is the question that matters: whether the *symbols* the wheel
    references allow the platform it claims, and whether the externals it
    cannot resolve are exactly those two — see :func:`platform_problem`.

    Args:
        wheel: The wheel to describe.

    Returns:
        The argument vector.
    """
    return ["auditwheel", "show", "--json", str(wheel)]


def platform_problem(report: Mapping[str, object]) -> str | None:
    """Report a wheel ``auditwheel show`` does not place on our platform.

    Args:
        report: Parsed output of :func:`show_arguments`.

    Returns:
        A message, or ``None`` when the wheel is what it says it is.
    """
    symbol_tag = report.get("sym_tag")
    if symbol_tag != PLATFORM_TAG:
        return (
            f"the versioned symbols this wheel references place it on "
            f"{symbol_tag!r}, not on {PLATFORM_TAG}. Every binary in it was "
            "compiled in the manylinux_2_34 image, so a lower tag means "
            "something was built against the host's glibc instead, and a "
            "higher one means the claimed tag is needlessly strict."
        )

    # Not `external_libs`: auditwheel computes that field against its most
    # compatible policy, whose whitelist is smaller than manylinux_2_34's, so
    # a library this platform does allow (libexpat, libmvec) would appear
    # there and read as a fault. `policy_upgrades` carries the per-policy
    # answer — what would have to be eliminated for *this* platform — and the
    # entry is absent entirely when nothing would.
    upgrades = report.get("policy_upgrades")
    entry = upgrades.get(PLATFORM_TAG, {}) if isinstance(upgrades, dict) else {}
    blocking = entry.get("libs_to_eliminate", []) if isinstance(entry, dict) else []
    unexpected = [soname for soname in blocking if not excluded(soname)]
    if unexpected:
        return (
            f"this wheel has unresolved external dependencies on "
            f"{', '.join(unexpected)}. The only libraries it may expect to "
            "find outside itself are the ones another wheel in the "
            f"environment owns ({', '.join(EXCLUDED_SONAMES)}); anything "
            "else has to be vendored, or the wheel fails to load on a "
            "machine that does not happen to have it."
        )
    return None


def abi3audit_arguments(wheel: Path) -> list[str]:
    """Return the ``abi3audit --strict`` command.

    Every extension in the wheel is built against the cp312 limited API
    (spec §4), and this is the gate that says so: ``--strict`` turns an
    extension that uses a symbol outside the limited API, or one whose abi3
    tag does not match what it really uses, into a failure.

    Args:
        wheel: The wheel to audit.

    Returns:
        The argument vector.
    """
    return ["abi3audit", "--strict", "--verbose", str(wheel)]


def is_elf(path: Path) -> bool:
    """Report whether a file is an ELF binary.

    Args:
        path: The file to look at.

    Returns:
        ``True`` when it starts with the ELF magic.
    """
    with path.open("rb") as opened:
        return opened.read(4) == b"\x7fELF"


def shipped_libraries(names: Iterable[str]) -> tuple[str, ...]:
    """Return the vendored shared libraries a wheel's member list carries.

    A library is a member whose file name looks like one: ``libpetsc.so.3.25``
    in the sibling directory, ``libgfortran-a1b2c3d4.so.5`` in auditwheel's
    graft directory, or anything similar a future layout puts elsewhere. The
    extension modules are deliberately not libraries here — ``cpp.abi3.so``
    and its two neighbours are this wheel's own build of DOLFINx and the
    bindings, covered by the entries for what they are rather than needing one
    of their own.

    Args:
        names: Members of the wheel.

    Returns:
        Their file names — not paths — sorted, so they can be checked against
        the notice file's component list wherever in the wheel they sit.
    """
    return tuple(
        sorted(
            Path(name).name
            for name in names
            if Path(name).name.startswith("lib") and ".so" in Path(name).name
        )
    )


def observe(wheel: Path, work_dir: Path, *, build_root: Path) -> Wheel:
    """Read what a built wheel says about itself.

    Args:
        wheel: The wheel to read.
        work_dir: Directory to unpack into, since the rpaths and the dynamic
            sections are read with the ELF tools.
        build_root: The container's build prefix, whose appearance in a text
            member is a leak.

    Returns:
        The facts :func:`wheel_problem` judges.
    """
    if work_dir.exists():
        shutil.rmtree(work_dir)
    unpack(wheel, work_dir)

    # The trailing separator makes this a search for the directory rather than
    # for the word: the README names `scripts/build-wheel.sh`, and the README
    # is part of METADATA.
    build_prefix = f"{build_root}/"
    names = tuple(
        sorted(
            path.relative_to(work_dir).as_posix()
            for path in work_dir.rglob("*")
            if path.is_file()
        )
    )
    runpaths = {}
    needed = {}
    build_paths = []
    for name in names:
        path = work_dir / name
        if is_elf(path):
            runpaths[name] = elf.read_runpath(path)
            needed[name] = elf.read_dynamic(path)[1]
        elif build_prefix in path.read_text(encoding="utf-8", errors="replace"):
            build_paths.append(name)

    tag = None
    wheel_file = work_dir / dist_info_name() / "WHEEL"
    if wheel_file.exists():
        for line in wheel_file.read_text(encoding="utf-8").splitlines():
            field, _, value = line.partition(":")
            if field.strip() == "Tag":
                tag = value.strip()

    return Wheel(
        names=names,
        tag=tag,
        libraries=shipped_libraries(names),
        runpaths=runpaths,
        needed=needed,
        build_paths=tuple(build_paths),
    )


def _tag_problem(built: Wheel) -> str | None:
    """Report a wheel that is not tagged the way spec §2 and §4 require."""
    if built.tag != WHEEL_TAG:
        return (
            f"this wheel declares Tag: {built.tag} rather than {WHEEL_TAG}. "
            "One wheel per release serves 3.12, 3.13 and 3.14 through the "
            "stable ABI, and the platform tag is what says the vendored stack "
            "was compiled against the right glibc (spec §2, §4)."
        )
    return None


def _member_problem(built: Wheel) -> str | None:
    """Report a member that must not be in a published wheel."""
    for name in built.names:
        parts = Path(name).parts
        if "__pycache__" in parts:
            return (
                f"{name} is bytecode compiled by the build's interpreter. The "
                "wheel is abi3 and installs on 3.12, 3.13 and 3.14, so "
                "shipping one interpreter's bytecode is at best dead weight."
            )
        if parts[0] == "basix":
            return (
                f"{name} is part of the fenics-basix wheel. The upstream trio "
                "is depended on and never re-vendored (spec §10); this is the "
                "staging symlink having been followed."
            )
    foreign = sorted(
        {
            parts[0]
            for name in built.names
            for parts in [Path(name).parts]
            if parts[0].endswith((".dist-info", ".egg-info"))
            and parts[0] != dist_info_name()
        }
    )
    if foreign:
        return (
            f"{', '.join(foreign)} is installed metadata for a distribution "
            "this wheel does not publish. pip refuses a wheel with more than "
            'one .dist-info outright — "invalid wheel, multiple .dist-info '
            "directories found\" — and registering someone else's "
            "distribution would be wrong even if it did not (spec §6, "
            "ticket 14)."
        )
    for patterns, message in REFUSED_LIBRARIES:
        found = [
            library
            for library in built.libraries
            if any(fnmatch(library, pattern) for pattern in patterns)
        ]
        if found:
            return f"this wheel carries {', '.join(found)}. {message}"
    if built.build_paths:
        return (
            f"{', '.join(built.build_paths)} still name the container's build "
            "prefix. A published wheel should not describe a directory that "
            "existed for an hour inside a container."
        )
    return None


def _payload_problem(built: Wheel) -> str | None:
    """Report a payload that is missing something the wheel needs."""
    expected = [
        NOTICES_PATH.as_posix(),
        "dolfinx_solver/__init__.py",
        "dolfinx/__init__.py",
        "dolfinx/cpp.abi3.so",
        "petsc4py/lib/PETSc.abi3.so",
        "slepc4py/lib/SLEPc.abi3.so",
        f"{dist_info_name()}/METADATA",
    ]
    missing = [name for name in expected if name not in built.names]
    if missing:
        return (
            f"this wheel is missing {', '.join(missing)}. Each of those is "
            "load-bearing: the notices are build-enforced (spec §7), the "
            "three extension modules are the product, and the METADATA is "
            "what brings the runtime stack — the PyPI mpich wheel above all — "
            "along with the install."
        )
    return None


def library_stem(name: str) -> str:
    """Return the library name without its version or auditwheel's hash.

    ``auditwheel`` appends its content hash to everything before the first dot
    of the file name — ``libpetsc.so.3.25`` is copied in as
    ``libpetsc-7dc9be3b.so.3.25.5`` and OpenBLAS's ``libopenblas-r0.3.32.so``
    as ``libopenblas-r0-0c4a605d.3.32.so`` — so that part, with a trailing
    eight-hex-digit suffix removed, is what a grafted copy and a vendored one
    still have in common.

    Args:
        name: File name of a shared library.

    Returns:
        Its stem, such as ``libpetsc`` for both spellings above.
    """
    stem = name.partition(".")[0]
    head, separator, suffix = stem.rpartition("-")
    if (
        separator
        and len(suffix) == 8
        and all(character in "0123456789abcdef" for character in suffix)
    ):
        return head
    return stem


def same_library(vendored: str, grafted: str) -> bool:
    """Report whether two stems name the same library.

    Usually they are equal. The exception is a library whose file name carries
    a version before the ``.so``, which the soname does not: OpenBLAS is
    vendored under its soname's stem, ``libopenblas``, and grafted under the
    file's, ``libopenblas-r0``. Matching on a hyphen boundary catches that
    without pairing ``libscotch`` with ``libscotcherr``.

    Args:
        vendored: Stem of a library in the sibling directory.
        grafted: Stem of a library in auditwheel's graft directory.

    Returns:
        ``True`` when the two are one library.
    """
    return vendored == grafted or grafted.startswith(f"{vendored}-")


def _duplicate_problem(built: Wheel) -> str | None:
    """Report a library the wheel carries twice.

    The vendored directory and auditwheel's graft directory holding the same
    library is not a waste of space, it is two libraries with two copies of
    whatever global state they keep — two PETSc option databases, two HDF5
    property-list registries — and which one a given caller reaches depends on
    the search path of the binary that asked. It happens when a staged library
    keeps an absolute ``RUNPATH`` into the build tree; see
    :func:`set_library_rpaths`.
    """
    vendored = {
        library_stem(Path(name).name)
        for name in built.names
        if Path(name).parent == bindings.VENDORED_LIBRARY_DIR
    }
    both = sorted(
        {
            (stem, Path(name).name)
            for name in built.names
            if Path(name).parts[0] == GRAFT_DIR
            for stem in vendored
            if same_library(stem, library_stem(Path(name).name))
        }
    )
    if both:
        return (
            "this wheel carries two copies of "
            + ", ".join(f"{stem} ({grafted})" for stem, grafted in both)
            + f": one in {bindings.VENDORED_LIBRARY_DIR} and one grafted into "
            f"{GRAFT_DIR}. auditwheel grafts what it resolves outside the "
            "wheel, so a second copy means a staged library's search path "
            "still points at the build tree rather than at $ORIGIN."
        )
    return None


def _licence_problem(built: Wheel) -> str | None:
    """Report a shipped library no component in the notice file accounts for."""
    unclaimed = notices.unclaimed_libraries(built.libraries)
    if unclaimed:
        return (
            f"{', '.join(unclaimed)} in this wheel are accounted for by no "
            "entry in wheelbuild.notices.COMPONENTS, so they would ship with "
            "no licence text beside them (spec §7). Either the build started "
            "producing a library nobody chose, or a new component needs its "
            "entry and its licence."
        )
    return None


def _rpath_problem(built: Wheel) -> str | None:
    """Report an rpath ``auditwheel repair`` rewrote into something wrong.

    The sibling layout and the reach into the basix wheel are the linkage
    design (spec §6, tickets 04 and 05), and repair preserves an existing
    entry only when it resolves inside the wheel — which is a rule about
    paths, not about intent. So the entries are read back and asserted.
    """
    expected = {
        "dolfinx/cpp.abi3.so": dolfinx.RELATIVE_RPATH_ENTRIES,
        f"{bindings.VENDORED_LIBRARY_DIR.as_posix()}/{dolfinx.soname()}": (
            dolfinx.LIBRARY_RPATH_ENTRIES
        ),
        "petsc4py/lib/PETSc.abi3.so": (bindings.RELATIVE_RPATH,),
        "slepc4py/lib/SLEPc.abi3.so": (bindings.RELATIVE_RPATH,),
    }
    for name, entries in expected.items():
        found = built.runpaths.get(name)
        if found is None:
            return (
                f"{name} is not an ELF file in this wheel, so the rpath the "
                "layout depends on cannot be checked."
            )
        for entry in entries:
            if entry not in found:
                return (
                    f"{name} no longer carries the rpath entry {entry} "
                    f"(it has {':'.join(found)}). That entry is how it "
                    "reaches the libraries this wheel vendors, or the ones "
                    "the fenics-basix wheel owns; auditwheel rewrites rpaths "
                    "and has dropped one the layout needs."
                )

    for name, found in sorted(built.runpaths.items()):
        absolute = [entry for entry in found if entry.startswith("/")]
        if absolute:
            return (
                f"{name} carries the absolute rpath entries "
                f"{', '.join(absolute)}. They resolve on the build machine "
                "and nowhere else, and one of them pointing at a directory "
                "that happens to exist on a user's machine is worse than one "
                "that does not."
            )
    return None


def _linkage_problem(built: Wheel) -> str | None:
    """Report the MPI linkage having been rewritten by the repair."""
    library_dir = bindings.VENDORED_LIBRARY_DIR.as_posix()
    petsc_library = f"{library_dir}/{petsc.soname()}"
    needed = built.needed.get(petsc_library)
    if needed is None:
        return (
            f"{petsc_library} is not an ELF file in this wheel, so the MPI "
            "linkage the whole design rests on cannot be checked. The "
            "vendored PETSc is what every solver call goes through, and a "
            "wheel without it under that name is not this wheel."
        )
    if not any(fnmatch(entry, "libmpi.so.*") for entry in needed):
        return (
            f"{petsc_library} no longer asks the loader for libmpi by its plain "
            "soname (it needs "
            f"{', '.join(sorted(entry for entry in needed if 'mpi' in entry))}). "
            "The excluded soname is what lets it bind to the copy mpi4py "
            "already loaded (spec §5, ADR-0002); a renamed one would bind to "
            "a grafted library instead."
        )

    for soname, removable in sorted(OVER_LINKED.items()):
        entries = built.needed.get(f"{library_dir}/{soname}")
        if entries is None:
            continue
        left = sorted(set(removable) & entries)
        if left:
            return (
                f"{soname} still needs {', '.join(left)}. Those are the "
                "dependencies MPICH's wrappers put on the link line and that "
                "nothing in the library calls; a wheel that keeps them needs "
                "UCX inside it (ADR-0003)."
            )
    return None


def wheel_problem(built: Wheel) -> str | None:
    """Report the first thing wrong with a built wheel.

    Args:
        built: What the wheel says about itself, from :func:`observe`.

    Returns:
        A message, or ``None`` when the wheel is publishable.
    """
    for check in (
        _tag_problem,
        _member_problem,
        _payload_problem,
        _duplicate_problem,
        _licence_problem,
        _rpath_problem,
        _linkage_problem,
    ):
        problem = check(built)
        if problem is not None:
            return problem
    return None


def validate_wheel(wheel: Path, work_dir: Path, *, build_root: Path) -> Wheel:
    """Read a built wheel and check everything the release depends on.

    Args:
        wheel: The wheel to check.
        work_dir: Directory to unpack into.
        build_root: The container's build prefix.

    Returns:
        What the wheel says about itself.

    Raises:
        ValueError: When a check fails; see :func:`wheel_problem`.
    """
    built = observe(wheel, work_dir, build_root=build_root)
    problem = wheel_problem(built)
    if problem is not None:
        raise ValueError(problem)
    return built


def install_arguments(wheel: Path, *, python: str) -> list[str]:
    """Return the command that installs the finished wheel.

    Args:
        wheel: The wheel to install.
        python: Interpreter of the clean venv to install into.

    Returns:
        The argument vector. Dependencies are resolved from the index, because
        the point of this install is that the metadata brings the runtime
        stack — the PyPI ``mpich`` wheel above all — with it (spec §5).
    """
    return [python, "-m", "pip", "install", "--quiet", str(wheel)]


def import_check_arguments(site: Path, *, python: str) -> list[str]:
    """Return the command that imports the installed wheel's stack.

    Args:
        site: The installed environment's ``site-packages``, which is the
            "staging site" the check asserts every package resolved under.
        python: Interpreter of that environment.

    Returns:
        The argument vector.
    """
    return [
        python,
        "-m",
        "wheelbuild.import_check",
        "--site",
        str(site),
        "--dolfinx",
    ]


def site_packages(python: str) -> Path:
    """Return where a venv's interpreter installs packages.

    Args:
        python: The interpreter.

    Returns:
        Its ``site-packages`` directory.

    Raises:
        subprocess.CalledProcessError: When the interpreter cannot be run.
    """
    return Path(
        capture(
            [
                python,
                "-c",
                (
                    "import sysconfig, sys; "
                    "sys.stdout.write(sysconfig.get_paths()['purelib'])"
                ),
            ]
        ).strip()
    )


def verify_install(wheel: Path, venv_dir: Path, *, base_python: str) -> Path:
    """Install the wheel into a clean venv and import the stack out of it.

    This is the acceptance criterion the rest of the assembly serves: the
    wheel installs on its own metadata, which brings the PyPI ``mpich`` wheel
    and the upstream trio with it, and the imports then prove the things only
    a live process shows — one MPI runtime, complex scalars, the rpaths
    resolving in the installed layout (:mod:`wheelbuild.import_check`).

    Args:
        wheel: The finished wheel.
        venv_dir: Directory for the venv. Replaced if it exists, since a
            reused environment is one where a previous wheel's files may
            still be sitting.
        base_python: Interpreter to make the venv from.

    Returns:
        The environment's ``site-packages``.
    """
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    check_call([base_python, "-m", "venv", str(venv_dir)])
    python = str(venv_dir / "bin" / "python")
    check_call([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    check_call(install_arguments(wheel, python=python))
    # The wheel brings its own runtime dependencies; this is the one thing the
    # *checker* needs and the wheel does not. wheelbuild.bindings imports
    # packaging to canonicalise distribution names, and the check runs the
    # driver modules inside this interpreter. It is installed after the wheel
    # so a missing runtime dependency still shows up as the wheel's own
    # install failing.
    check_call([python, "-m", "pip", "install", "--quiet", *CHECK_REQUIREMENTS])

    site = site_packages(python)
    # LD_LIBRARY_PATH is dropped for the same reason the audits drop it: with
    # the build prefix on it the installed wheel's libraries would resolve to
    # the tree they were built in rather than through the rpaths they carry,
    # and the check would pass for a wheel that cannot work anywhere else.
    # The driver modules are imported from the repository, as everywhere else
    # in this build.
    environment = audit_environment()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    check_call(import_check_arguments(site, python=python), env=environment)
    return site


def run(
    *,
    prefix: Path,
    build_root: Path,
    wheelhouse: Path,
    image_licenses: Path = Path("/usr/share/licenses"),
    python: str = sys.executable,
    base_python: str | None = None,
) -> Path:
    """Assemble, repair, audit and prove the wheel.

    Args:
        prefix: The shared install prefix every earlier stage installed into.
        build_root: Scratch root, holding the source trees the notices are
            harvested from and the directories this step works in.
        wheelhouse: Where the finished wheel is left.
        image_licenses: The build image's licence directory.
        python: Interpreter to build the base wheel with — the build venv's.
        base_python: Interpreter the clean test venv is made from. Defaults to
            ``python``.

    Returns:
        The finished wheel.

    Raises:
        FileNotFoundError: When an input is missing.
        ValueError: When the wheel is not publishable; see
            :func:`wheel_problem`.
    """
    work = build_root / "assemble"
    work.mkdir(parents=True, exist_ok=True)

    base = source_copy(work / "source")
    base_wheelhouse = work / "base"
    if base_wheelhouse.exists():
        shutil.rmtree(base_wheelhouse)
    check_call(base_wheel_arguments(base, base_wheelhouse, python=python))
    base_wheels = sorted(base_wheelhouse.glob(f"{DISTRIBUTION}-*.whl"))
    if len(base_wheels) != 1:
        raise FileNotFoundError(
            f"expected one base wheel in {base_wheelhouse}, found "
            f"{len(base_wheels)}. It is built from a copy of the repository's "
            "own package and is the metadata half of the artefact."
        )

    staging = stage(
        prefix=prefix,
        base_wheel=base_wheels[0],
        staging=work / "wheel",
        build_root=build_root,
        image_licenses=image_licenses,
    )
    assembled = pack(staging, work / wheel_name(BUILD_TAG))

    wheelhouse.mkdir(parents=True, exist_ok=True)
    repaired = wheelhouse / wheel_name(WHEEL_TAG)
    if repaired.exists():
        repaired.unlink()
    environment = audit_environment()
    check_call(repair_arguments(assembled, wheelhouse), env=environment)
    if not repaired.exists():
        raise FileNotFoundError(
            f"auditwheel repair left no {repaired.name} in {wheelhouse}: the "
            "vendored stack did not qualify for the platform this wheel "
            f"claims ({PLATFORM_TAG})."
        )

    report = json.loads(capture(show_arguments(repaired), env=environment))
    problem = platform_problem(report)
    if problem is not None:
        raise ValueError(problem)
    check_call(abi3audit_arguments(repaired))
    validate_wheel(repaired, work / "validate", build_root=build_root)
    verify_install(repaired, work / "venv", base_python=base_python or python)
    return repaired


def existing_wheel(wheelhouse: Path) -> Path:
    """Return the finished wheel a warm cache already left in the wheelhouse.

    Args:
        wheelhouse: Directory the assembly writes the wheel into.

    Returns:
        The wheel.

    Raises:
        FileNotFoundError: When it is not there.
    """
    wheel = wheelhouse / wheel_name(WHEEL_TAG)
    if not wheel.exists():
        raise FileNotFoundError(
            f"no wheel at {wheel}, so there is nothing to re-check. The "
            "assembly is cheap compared with the compile it follows; run it "
            "without --validate-only."
        )
    return wheel


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the wheel assembly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument(
        "--image-licenses", type=Path, default=Path("/usr/share/licenses")
    )
    parser.add_argument(
        "--base-python",
        help="interpreter the clean test venv is made from; defaults to this one",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check an already-built wheel instead of assembling one",
    )
    args = parser.parse_args(argv)

    try:
        if args.validate_only:
            wheel = existing_wheel(args.wheelhouse)
            validate_wheel(
                wheel,
                args.build_root / "assemble" / "validate",
                build_root=args.build_root,
            )
        else:
            wheel = run(
                prefix=args.prefix,
                build_root=args.build_root,
                wheelhouse=args.wheelhouse,
                image_licenses=args.image_licenses,
                base_python=args.base_python,
            )
    except (FileNotFoundError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"{wheel.name} assembled: {len(notices.COMPONENTS)} components "
        f"noticed, tagged {WHEEL_TAG}, imported out of a clean venv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
