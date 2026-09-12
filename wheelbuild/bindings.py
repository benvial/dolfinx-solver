"""Build our own petsc4py and slepc4py against the vendored PETSc and SLEPc.

The wheel ships both bindings under their upstream import names and under no
distribution name at all (spec §6). That is the conda model: one environment,
one petsc4py, built against the one complex PETSc this wheel carries. The PyPI
projects of those names are never a dependency and never installed beside
ours — ``dolfinx_solver._bootstrap`` refuses an environment where they are —
so the staged tree here holds ``petsc4py/`` and ``slepc4py/`` packages with
their ``.dist-info`` directories stripped off.

Three things make that work, and each is checked rather than assumed:

* **The sources ride in the tarballs the build already downloaded.**
  ``petsc-3.25.5/src/binding/petsc4py`` and
  ``slepc-3.25.1/src/binding/slepc4py`` are the bindings for exactly the
  libraries in the prefix, which is a coupling no version pin could express as
  precisely.
* **The extension is limited-API.** Both ``setup.py`` files read a
  ``*_BUILD_PYSABI`` environment variable and, when it names an interpreter,
  compile with ``Py_LIMITED_API`` and let ``bdist_wheel`` tag the result. The
  proof is the file name the build leaves behind: ``PETSc.abi3.so`` rather
  than ``PETSc.cpython-312-x86_64-linux-gnu.so``, which is what lets one wheel
  serve 3.12, 3.13 and 3.14 (spec §4).
* **The extension finds ``libpetsc`` by a relative path.** Upstream's
  ``conf/confpetsc.py`` writes ``$ORIGIN/../../petsc/lib`` only when the PyPI
  ``petsc`` package is what supplies the library; for everything else it bakes
  in the absolute ``PETSC_LIB_DIR`` — here ``/build/install/lib``, a directory
  that exists on no user's machine. The rpath is therefore rewritten after the
  build to :data:`RELATIVE_RPATH`, the same ``$ORIGIN`` pattern aimed at where
  the wheel actually keeps its libraries.

The staging directory is what makes that last point provable before the wheel
assembly step exists: it is laid out as the wheel is, with the vendored
library directory beside the bindings, so the import check can run with no
``LD_LIBRARY_PATH`` to lean on. If the relative rpath is wrong, the import
fails here rather than on a user's machine.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from wheelbuild import elf, petsc, slepc
from wheelbuild._process import check_call

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Repository root, which is where the driver modules are imported from. The
#: import check runs in a fresh interpreter and needs both this and the staged
#: packages on its path.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The interpreter the abi3 wheel targets (spec §4). Passed to both bindings'
#: ``*_BUILD_PYSABI`` variables, which turn it into ``Py_LIMITED_API`` and the
#: ``cp312-abi3`` wheel tag.
LIMITED_API_TAG = "cp312"

#: Where the wheel keeps its vendored shared libraries, relative to the wheel
#: root. The bindings sit beside ``dolfinx_solver`` as top-level packages, so
#: this is the sibling directory their rpath climbs to.
VENDORED_LIBRARY_DIR = Path("dolfinx_solver/lib")

#: Name of the staging directory inside the install prefix: a stand-in for the
#: wheel root, holding what the assembly step (ticket 06) will graft into the
#: wheel as top-level packages.
SITE_DIR_NAME = "python"

#: The rpath written onto both extensions, replacing the absolute build path
#: upstream's ``setup.py`` bakes in. An extension lives two directories down
#: from the wheel root (``petsc4py/lib/PETSc.abi3.so``), so two hops up and
#: back down into the vendored library directory is the sibling layout spec §6
#: asks for.
RELATIVE_RPATH = "$ORIGIN/../../" + VENDORED_LIBRARY_DIR.as_posix()


class Binding(NamedTuple):
    """One of the two Python bindings this wheel builds for itself.

    Attributes:
        import_name: Upstream import name, which is also the package directory
            in the staged tree and the distribution name we never publish.
        module_name: The extension module inside it.
        source_relative: Where its sources sit inside the tarball of the
            library it binds.
        sabi_variable: The environment variable its ``setup.py`` reads to
            decide whether to build against the limited API.
        install_dir_variables: The environment variables its build reads the
            install prefix from. slepc4py needs both halves of the stack.
        library_soname: The vendored library the finished extension has to
            ask the loader for.
    """

    import_name: str
    module_name: str
    source_relative: Path
    sabi_variable: str
    install_dir_variables: tuple[str, ...]
    library_soname: str


#: petsc4py, out of the PETSc tarball, against the vendored complex PETSc.
PETSC4PY = Binding(
    import_name="petsc4py",
    module_name="PETSc",
    source_relative=Path("src/binding/petsc4py"),
    sabi_variable="PETSC4PY_BUILD_PYSABI",
    install_dir_variables=("PETSC_DIR",),
    library_soname=petsc.soname(),
)

#: slepc4py, out of the SLEPc tarball. SLEPc was configured
#: ``--with-slepc4py=0`` deliberately (ticket 03), so this is built here
#: against the finished install rather than as a by-product of that build.
SLEPC4PY = Binding(
    import_name="slepc4py",
    module_name="SLEPc",
    source_relative=Path("src/binding/slepc4py"),
    sabi_variable="SLEPC4PY_BUILD_PYSABI",
    install_dir_variables=("PETSC_DIR", "SLEPC_DIR"),
    library_soname=slepc.soname(),
)

#: Both, in the order they have to be built: slepc4py's ``setup.py`` imports
#: petsc4py for its include directory.
BINDINGS = (PETSC4PY, SLEPC4PY)


class Build(NamedTuple):
    """What a staged extension module says about itself.

    Attributes:
        runpath: Its ``DT_RUNPATH`` directories, where the absolute build path
            would show up.
        needed: Its ``DT_NEEDED`` entries, where the binding to the vendored
            library shows up.
    """

    runpath: tuple[str, ...]
    needed: frozenset[str]


def source_dir(binding: Binding, tarball_dir: Path) -> Path:
    """Return where a binding's sources sit inside an unpacked tarball.

    Args:
        binding: The binding being built.
        tarball_dir: Unpacked source tree of the library it binds.

    Returns:
        The directory holding its ``setup.py``.
    """
    return tarball_dir / binding.source_relative


def extension_path(binding: Binding) -> Path:
    """Return where a binding's extension module lands, relative to the site.

    Args:
        binding: The binding being built.

    Returns:
        The path, such as ``petsc4py/lib/PETSc.abi3.so``. The ``abi3`` in the
        name is the limited-API build's own record of itself: setuptools names
        the file that way only when the extension was compiled against the
        stable ABI.
    """
    return Path(binding.import_name) / "lib" / f"{binding.module_name}.abi3.so"


def required_artefacts(binding: Binding) -> tuple[Path, ...]:
    """Return what a finished staging tree has to contain for one binding.

    The vendored library is on the list because the rpath is relative: the
    extension reaches it through the staging tree, so an extension without it
    beside is an import that fails at load time.

    Args:
        binding: The binding being built.

    Returns:
        The paths, relative to the staging site, in declaration order.
    """
    return (
        Path(binding.import_name) / "__init__.py",
        extension_path(binding),
        VENDORED_LIBRARY_DIR / binding.library_soname,
    )


def site_dir(prefix: Path) -> Path:
    """Return the staging site inside an install prefix.

    Args:
        prefix: The shared install prefix.

    Returns:
        The directory the wheel payload is staged in.
    """
    return prefix / SITE_DIR_NAME


def prepare_site(prefix: Path) -> Path:
    """Lay the staging site out the way the wheel is laid out.

    The vendored library directory is a symlink to the prefix's ``lib`` rather
    than a copy: the libraries are hundreds of megabytes and the assembly step
    is what really places them. What matters here is that
    :data:`RELATIVE_RPATH` resolves, so the import check proves the rpath
    instead of proving whatever ``LD_LIBRARY_PATH`` happened to hold.

    Args:
        prefix: The shared install prefix. Its ``lib`` is what the link points
            at.

    Returns:
        The staging site, ready for packages to be unpacked into.

    Raises:
        NotADirectoryError: When the library directory is a real directory,
            which would mean something else already staged libraries there.
    """
    site = site_dir(prefix)
    link = site / VENDORED_LIBRARY_DIR
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise NotADirectoryError(
            f"{link} is a real directory, not the link into {prefix / 'lib'} "
            "this stage makes. The staging site holds packages and one "
            "symlink; vendoring the libraries themselves is the assembly "
            "step's job."
        )
    link.symlink_to(
        os.path.relpath(prefix / "lib", link.parent), target_is_directory=True
    )
    return site


def build_environment(
    binding: Binding,
    *,
    prefix: Path,
    site: Path,
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the environment a binding's build needs.

    ``PETSC_ARCH`` is set to the empty string for the reason the SLEPc driver
    gives: the install is a prefix install and has no arch directory, and an
    inherited value would send the build looking inside one.

    Bytecode writing is turned off because slepc4py's ``setup.py`` imports the
    staged petsc4py for its include directory, and the staging site is the
    wheel's payload: a ``__pycache__`` written into it during the build is a
    directory the assembly step would graft into the wheel.

    Args:
        binding: The binding being built.
        prefix: The shared install prefix, holding PETSc and SLEPc.
        site: The staging site, which is where an already-built petsc4py is
            imported from when slepc4py's ``setup.py`` asks for its includes.
        environ: Environment to extend. Defaults to this process's.

    Returns:
        A complete environment for the build.
    """
    return {
        **(os.environ if environ is None else environ),
        **{variable: str(prefix) for variable in binding.install_dir_variables},
        "PETSC_ARCH": "",
        binding.sabi_variable: LIMITED_API_TAG,
        "PYTHONPATH": str(site),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def wheel_arguments(
    *, source_dir: Path, wheelhouse: Path, python: str = sys.executable
) -> list[str]:
    """Return the command that builds one binding's wheel.

    Nothing is downloaded: the build dependencies (setuptools, Cython, numpy)
    are installed in the container's build venv, so ``--no-build-isolation``
    keeps the build using the versions this build controls, and ``--no-deps``
    keeps pip from resolving the PyPI ``petsc4py`` the wheel must never carry.

    Args:
        source_dir: The binding's source directory.
        wheelhouse: Where to leave the built wheel.
        python: Interpreter to run pip from — the build venv's.

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


def install_arguments(wheel: Path, *, python: str = sys.executable) -> list[str]:
    """Return the command that installs a built binding into the build venv.

    The wheel payload comes from the staging site, not from here. This install
    exists so the *next* build step can import the binding: slepc4py's
    ``setup.py`` needs petsc4py's include directory and names petsc4py in
    ``setup_requires``, which setuptools resolves against installed
    distribution metadata and would otherwise fetch from PyPI. DOLFINx's
    nanobind casters (ticket 05) need the same includes.

    Args:
        wheel: The built wheel.
        python: Interpreter to run pip from — the build venv's.

    Returns:
        The argument vector.
    """
    return [
        python,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--force-reinstall",
        str(wheel),
    ]


def built_wheel(wheelhouse: Path, binding: Binding) -> Path:
    """Return the wheel a binding's build just left in the wheelhouse.

    Args:
        wheelhouse: Directory the build writes wheels into.
        binding: The binding that was built.

    Returns:
        The newest wheel of that binding.

    Raises:
        FileNotFoundError: When the build left none.
    """
    wheels = sorted(
        wheelhouse.glob(f"{binding.import_name}-*.whl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not wheels:
        raise FileNotFoundError(
            f"no {binding.import_name} wheel in {wheelhouse}: the build "
            "reported success and produced nothing."
        )
    return wheels[-1]


def wheel_tag_problem(name: str) -> str | None:
    """Report a wheel that is not the limited-API one.

    Args:
        name: File name of the built wheel.

    Returns:
        A message, or ``None`` when the wheel is tagged ``cp312-abi3``.
    """
    expected = f"-{LIMITED_API_TAG}-abi3-"
    if expected in name:
        return None
    return (
        f"{name} is not tagged {expected.strip('-')}: the build ignored the "
        "limited-API request, so this extension is bound to one interpreter "
        "and the wheel it rides in could not serve 3.12, 3.13 and 3.14 alike "
        "(spec §4)."
    )


def unpack(wheel: Path, site: Path, binding: Binding) -> Path:
    """Unpack a built binding's wheel into the staging site.

    The ``.dist-info`` directory is dropped on purpose. Keeping it would
    install a distribution called ``petsc4py`` inside our wheel, which is
    exactly what ``dolfinx_solver._bootstrap`` refuses as a foreign PETSc
    stack — the wheel ships the import names and no distribution of those
    names (spec §6).

    Args:
        wheel: The built wheel.
        site: The staging site.
        binding: The binding being staged.

    Returns:
        The staged package directory.
    """
    package = site / binding.import_name
    if package.exists():
        shutil.rmtree(package)

    print(f"+ unpack {wheel} into {site}", flush=True)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            top = Path(member).parts[0]
            if top.endswith((".dist-info", ".data")):
                continue
            archive.extract(member, site)

    # Zip files carry no permissions the extractor keeps, and a shared object
    # is read by the loader, not executed — but the wheel's own libraries are
    # executable and the payload should not be the odd one out.
    for library in package.rglob("*.so"):
        library.chmod(0o755)
    return package


def patchelf_arguments(extension: Path) -> list[str]:
    """Return the command that rewrites an extension's rpath.

    Args:
        extension: The built extension module, in the staging site.

    Returns:
        The argument vector, replacing whatever absolute build path upstream's
        ``setup.py`` baked in with :data:`RELATIVE_RPATH`.
    """
    return ["patchelf", "--set-rpath", RELATIVE_RPATH, str(extension)]


def observe(binding: Binding, site: Path) -> Build:
    """Read what a staged extension says about itself.

    Args:
        binding: The binding being checked.
        site: The staging site.

    Returns:
        The facts :func:`build_problem` judges.

    Raises:
        subprocess.CalledProcessError: When ``readelf`` cannot read the
            extension.
    """
    extension = site / extension_path(binding)
    _, needed = elf.read_dynamic(extension)
    return Build(runpath=elf.read_runpath(extension), needed=needed)


def build_problem(binding: Binding, build: Build) -> str | None:
    """Report an extension that would not find its library in a wheel.

    Args:
        binding: The binding being checked.
        build: What the staged extension says about itself.

    Returns:
        A message naming the first thing that is wrong, or ``None``.
    """
    if RELATIVE_RPATH not in build.runpath:
        found = ", ".join(build.runpath) or "nothing"
        return (
            f"{extension_path(binding)} looks for its libraries in {found} "
            f"rather than {RELATIVE_RPATH}. Upstream's setup.py bakes in the "
            "absolute PETSC_LIB_DIR of the machine that built it, which is a "
            "directory no user has; the rpath is rewritten to the wheel's own "
            "sibling layout after the build (spec §6)."
        )

    absolute = [entry for entry in build.runpath if not entry.startswith("$ORIGIN")]
    if absolute:
        return (
            f"{extension_path(binding)} still carries the build path(s) "
            f"{', '.join(absolute)} on its rpath. A path that exists only in "
            "the build container either resolves to nothing on a user's "
            "machine or, worse, to somebody else's PETSc."
        )

    if binding.library_soname not in build.needed:
        found = ", ".join(sorted(build.needed)) or "nothing"
        return (
            f"{extension_path(binding)} does not ask the loader for "
            f"{binding.library_soname} (it needs {found}). This binding is "
            "built against the library this wheel vendors, and an extension "
            "that names another one is a build that found something else on "
            "the machine."
        )
    return None


def missing_artefacts(binding: Binding, site: Path) -> list[Path]:
    """Return the required artefacts a staging site does not have.

    Args:
        binding: The binding being checked.
        site: The staging site.

    Returns:
        The missing paths, relative to the site, in declaration order.
    """
    return [
        relative
        for relative in required_artefacts(binding)
        if not (site / relative).exists()
    ]


def distribution_artefacts(site: Path) -> list[Path]:
    """Return the installed-distribution metadata a staging site carries.

    Args:
        site: The staging site.

    Returns:
        The ``.dist-info`` and ``.egg-info`` directories found, sorted. There
        must be none: see :func:`unpack`.
    """
    return sorted(
        path.relative_to(site)
        for pattern in ("*.dist-info", "*.egg-info")
        for path in site.glob(pattern)
    )


def validate(binding: Binding, site: Path, build: Build) -> Path:
    """Check a staged binding against everything the wheel needs to be true.

    Args:
        binding: The binding being checked.
        site: The staging site.
        build: What the staged extension says about itself.

    Returns:
        The validated staging site.

    Raises:
        FileNotFoundError: When the staging tree is incomplete.
        ValueError: When it carries an upstream distribution, or when the
            extension would not find its library in a wheel.
    """
    missing = missing_artefacts(binding, site)
    if missing:
        raise FileNotFoundError(
            f"incomplete {binding.import_name} staging tree at {site}: "
            "missing " + ", ".join(str(relative) for relative in missing)
        )

    distributions = distribution_artefacts(site)
    if distributions:
        carried = ", ".join(str(relative) for relative in distributions)
        raise ValueError(
            f"the staging site carries installed-distribution metadata "
            f"({carried}). The wheel ships petsc4py and slepc4py under their "
            "import names and under no distribution name: a dist-info in the "
            "payload registers a petsc4py distribution, which is precisely "
            "what dolfinx_solver refuses at import as a foreign PETSc stack."
        )

    problem = build_problem(binding, build)
    if problem is not None:
        raise ValueError(problem)
    return site


def validate_install(binding: Binding, site: Path) -> Path:
    """Read the facts about a staged binding, then :func:`validate` them.

    Args:
        binding: The binding being checked.
        site: The staging site.

    Returns:
        The validated staging site.

    Raises:
        FileNotFoundError: When the staging tree is incomplete.
        ValueError: When a check fails; see :func:`validate`.
    """
    missing = missing_artefacts(binding, site)
    if missing:
        raise FileNotFoundError(
            f"incomplete {binding.import_name} staging tree at {site}: "
            "missing " + ", ".join(str(relative) for relative in missing)
        )
    return validate(binding, site, observe(binding, site))


def import_check_arguments(*, site: Path, python: str = sys.executable) -> list[str]:
    """Return the command that imports the staged bindings in a fresh process.

    Args:
        site: The staging site the bindings are imported from.
        python: Interpreter to run — the build venv's, which has mpi4py and
            the PyPI ``mpich`` wheel's ``libmpi``.

    Returns:
        The argument vector.
    """
    return [python, "-m", "wheelbuild.import_check", "--site", str(site)]


def import_check_environment(
    *, site: Path, environ: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the environment the import check runs in.

    ``LD_LIBRARY_PATH`` is removed rather than left alone. The container build
    puts the install prefix on it so each stage can link against the one
    before, and leaving it set would resolve ``libpetsc`` for the extension
    whatever its rpath said — turning the one check that proves the wheel's
    linkage into a check that proves the build container's.

    Bytecode writing is turned off for a different reason: the staging site is
    the wheel's payload, and an interpreter importing out of it leaves
    ``__pycache__`` directories that the assembly step would otherwise graft
    into the wheel.

    Args:
        site: The staging site, which goes first on the path so the staged
            packages win over anything installed in the venv.
        environ: Environment to base this on. Defaults to this process's.

    Returns:
        A complete environment for the import check.
    """
    environment = dict(os.environ if environ is None else environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment["PYTHONPATH"] = os.pathsep.join([str(site), str(REPO_ROOT)])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def build_one(
    binding: Binding,
    *,
    tarball_dir: Path,
    prefix: Path,
    site: Path,
    wheelhouse: Path,
    python: str = sys.executable,
) -> Path:
    """Build, stage, re-rpath and validate one binding.

    Args:
        binding: The binding to build.
        tarball_dir: Unpacked source tree of the library it binds.
        prefix: The shared install prefix.
        site: The staging site.
        wheelhouse: Where built wheels are left.
        python: Interpreter to run pip from.

    Returns:
        The validated staging site.

    Raises:
        ValueError: When the build is not limited-API, or when the staged
            extension would not find its library in a wheel.
    """
    check_call(
        wheel_arguments(
            source_dir=source_dir(binding, tarball_dir),
            wheelhouse=wheelhouse,
            python=python,
        ),
        env=build_environment(binding, prefix=prefix, site=site),
    )
    wheel = built_wheel(wheelhouse, binding)

    problem = wheel_tag_problem(wheel.name)
    if problem is not None:
        raise ValueError(problem)

    unpack(wheel, site, binding)
    check_call(patchelf_arguments(site / extension_path(binding)))
    check_call(install_arguments(wheel, python=python))
    return validate_install(binding, site)


def run(
    *,
    petsc_source: Path,
    slepc_source: Path,
    prefix: Path,
    python: str = sys.executable,
) -> Path:
    """Build both bindings and prove they import together.

    Args:
        petsc_source: Unpacked PETSc source tree, holding petsc4py's sources.
        slepc_source: Unpacked SLEPc source tree, holding slepc4py's.
        prefix: The shared install prefix.
        python: Interpreter to build and check with — the build venv's.

    Returns:
        The staging site holding both bindings.
    """
    site = prepare_site(prefix)
    wheelhouse = prefix / "wheelhouse"
    wheelhouse.mkdir(parents=True, exist_ok=True)
    tarballs = {PETSC4PY: petsc_source, SLEPC4PY: slepc_source}

    for binding in BINDINGS:
        build_one(
            binding,
            tarball_dir=tarballs[binding],
            prefix=prefix,
            site=site,
            wheelhouse=wheelhouse,
            python=python,
        )

    return check_imports(site, python=python)


def check_imports(site: Path, *, python: str = sys.executable) -> Path:
    """Import both staged bindings in a fresh interpreter and check the result.

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
        env=import_check_environment(site=site),
    )
    return site


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the bindings build and its validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check an already-staged site instead of building one",
    )
    parser.add_argument("--petsc-source", type=Path)
    parser.add_argument("--slepc-source", type=Path)
    args = parser.parse_args(argv)
    if not args.validate_only and (
        args.petsc_source is None or args.slepc_source is None
    ):
        parser.error("--petsc-source and --slepc-source are required to build")

    try:
        if args.validate_only:
            site = site_dir(args.prefix)
            for binding in BINDINGS:
                validate_install(binding, site)
            check_imports(site)
        else:
            site = run(
                petsc_source=args.petsc_source,
                slepc_source=args.slepc_source,
                prefix=args.prefix,
            )
    except (FileNotFoundError, NotADirectoryError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"petsc4py {petsc.PETSC_VERSION} and slepc4py {slepc.SLEPC_VERSION} "
        f"staged in {site}, {LIMITED_API_TAG}-abi3, reaching "
        f"{petsc.soname()} and {slepc.soname()} through {RELATIVE_RPATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
