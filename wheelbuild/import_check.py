"""Import the staged bindings the way a user's process will, and check it.

Everything else in the build reads files. This is the one step that runs the
stack, because three of the wheel's load-time properties exist only in a live
process and cannot be read off an ELF header:

* **One MPI runtime.** ``mpi4py.MPI`` is imported first, its ``_mpiabi``
  dlopens ``libmpi.so.12`` out of the environment prefix, and every later
  ``DT_NEEDED`` binds to that already-loaded object by soname (spec §5,
  ADR-0002). The install prefix contains a second ``libmpi.so.12`` — the one
  the MPICH stage built and the wheel discards — so this is not a theoretical
  risk here: ``/proc/self/maps`` says which copies the process actually
  mapped, and two of them would mean the import order stopped working.
* **Complex scalars.** ``PetscScalar`` is baked into ``libpetsc``, into the
  extension and later into DOLFINx's binaries alike; ``PETSc.ScalarType`` is
  what the built stack reports it to be.
* **The staged bindings, not somebody else's.** The build venv has petsc4py
  installed as a distribution, because slepc4py's ``setup.py`` and DOLFINx's
  build need it importable. The wheel's copy is the one in the staging site,
  so the check asserts where each binding actually resolved.

The imports go through an injected ``import_module`` for the reason
:mod:`dolfinx_solver._bootstrap` does the same: the ordering is the thing
being checked, and it can then be checked without a built stack.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wheelbuild import bindings, mpich

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: The module whose import has to come first, and the whole mechanism by
#: which the wheel and mpi4py share one MPI runtime.
MPI_MODULE = "mpi4py.MPI"

#: Where the kernel reports the libraries this process has mapped.
MAPS_PATH = Path("/proc/self/maps")


def mapped_libraries(maps_text: str, soname: str) -> set[str]:
    """Return the distinct files with a given soname mapped into a process.

    A loaded library appears once per segment, and the file it was opened from
    may carry the release in its name (``libmpi.so.12.6.1``) or be the soname
    itself, depending on which copy the loader found. Both spellings are the
    same library; two different paths are two libraries.

    Args:
        maps_text: Contents of ``/proc/self/maps``.
        soname: The soname to look for, such as ``libmpi.so.12``.

    Returns:
        The paths, one per distinct file.
    """
    found = set()
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5].strip()
        if not path.startswith("/"):
            continue
        name = path.rpartition("/")[2]
        if name == soname or name.startswith(f"{soname}."):
            found.add(path)
    return found


def mpi_problem(maps_text: str) -> str | None:
    """Report a process that has not got exactly one MPI runtime in it.

    Args:
        maps_text: Contents of ``/proc/self/maps``, read after the imports.

    Returns:
        A message, or ``None`` when one ``libmpi`` is mapped.
    """
    mapped = mapped_libraries(maps_text, mpich.MPI_SONAME)
    if not mapped:
        return (
            f"no {mpich.MPI_SONAME} is mapped into this process at all, so "
            "either mpi4py did not load one or it loaded it under another "
            "name. The wheel's whole MPI linkage rests on that library being "
            "in the process before any compiled module (spec §5)."
        )
    if len(mapped) > 1:
        return (
            f"{len(mapped)} MPI runtimes are mapped into this process: "
            f"{', '.join(sorted(mapped))}. mpi4py's copy is the one every "
            "vendored library has to bind to; a second one means two MPI "
            "global states, which shows up as a hang or a wrong answer "
            "rather than an error (spec §5, ADR-0002)."
        )
    return None


def scalar_problem(scalar_type: Any) -> str | None:
    """Report a ``PetscScalar`` that is not complex.

    Args:
        scalar_type: ``PETSc.ScalarType``, the numpy scalar type the built
            PETSc reports.

    Returns:
        A message, or ``None`` when the type holds an imaginary part.
    """
    try:
        value = scalar_type(1j)
    except TypeError:
        return (
            f"PETSc.ScalarType is {getattr(scalar_type, '__name__', scalar_type)}, "
            "which cannot hold a complex number. This is the complex-scalar "
            "distribution; a real build belongs to dolfinx-solver-real, which "
            "is a different wheel because the scalar type is baked into every "
            "binary in it (spec §6)."
        )
    if value.imag != 1:
        return (
            f"PETSc.ScalarType({value!r}) dropped the imaginary part, so this "
            "stack is not the complex-scalar one this distribution ships."
        )
    return None


def origin_problem(import_name: str, module: Any, site: Path) -> str | None:
    """Report a binding that was imported from outside the staging site.

    Args:
        import_name: The binding's import name.
        module: The module that was imported.
        site: The staging site holding the wheel's payload.

    Returns:
        A message, or ``None`` when the import resolved into the site.
    """
    origin = getattr(module, "__file__", None)
    if origin is not None and site.resolve() in Path(origin).resolve().parents:
        return None
    return (
        f"{import_name} was imported from {origin}, which is not the copy "
        f"staged in {site}. The build venv has petsc4py installed so that "
        "slepc4py and DOLFINx can compile against it; the wheel's own build "
        "is the staged one, and it is the one this check has to exercise."
    )


def check(
    *,
    site: Path,
    import_module: Callable[[str], Any] = importlib.import_module,
    maps_text: str | None = None,
) -> str | None:
    """Import the stack in order and report the first thing that is wrong.

    Args:
        site: The staging site the bindings are imported from.
        import_module: How to import a module by name. Injected so the
            ordering can be checked without a built stack.
        maps_text: Contents of ``/proc/self/maps``. Read after the imports
            when not given, which is what a real run does.

    Returns:
        A message naming the first problem, or ``None`` when the stack holds
        together.
    """
    try:
        import_module(MPI_MODULE)
    except ImportError as error:
        return (
            f"importing {MPI_MODULE} failed: {error}. It is what dlopens the "
            "PyPI mpich wheel's libmpi before any compiled module of ours, so "
            "nothing further in this check would mean anything. The build "
            "venv needs mpi4py and the mpich wheel installed, and this check "
            "runs without LD_LIBRARY_PATH on purpose."
        )

    modules = {}
    for binding in bindings.BINDINGS:
        name = f"{binding.import_name}.{binding.module_name}"
        try:
            modules[binding.import_name] = import_module(name)
        except ImportError as error:
            return (
                f"importing {name} out of {site} failed: {error}. The "
                "extension is built against the vendored libraries and finds "
                f"them through {bindings.RELATIVE_RPATH}, so a missing "
                "library here is an rpath that does not resolve in the "
                "wheel's layout."
            )
        problem = origin_problem(
            binding.import_name, modules[binding.import_name], site
        )
        if problem is not None:
            return problem

    problem = scalar_problem(modules[bindings.PETSC4PY.import_name].ScalarType)
    if problem is not None:
        return problem

    if maps_text is None:
        maps_text = MAPS_PATH.read_text(encoding="utf-8")
    return mpi_problem(maps_text)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the in-process import check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args(argv)

    problem = check(site=args.site)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    mapped = sorted(
        mapped_libraries(MAPS_PATH.read_text(encoding="utf-8"), mpich.MPI_SONAME)
    )
    print(
        f"petsc4py and slepc4py imported from {args.site} after {MPI_MODULE}, "
        f"complex scalars, one MPI runtime ({mapped[0]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
