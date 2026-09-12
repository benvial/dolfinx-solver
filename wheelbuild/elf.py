"""Read what a shared library declares: its symbols and its dynamic section.

The wheel's MPI linkage is split across two halves that are never built
together — the ``libmpifort`` compiled in this container and the
``libmpi.so.12`` a user's ``pip install`` fetches from the PyPI ``mpich``
wheel. Nothing at install time checks that the second can satisfy the first:
the soname is frozen across MPICH series, so the loader binds them happily and
any mismatch surfaces as an undefined-symbol error at the user's first import
(ADR-0001).

Comparing the two declarations in the build is what turns that into a build
failure instead. Everything here reads what a file on disk says about itself,
rather than running a binary and asking: a built ``mpichversion`` reports
whichever ``libmpi`` its own loader resolves, which on a machine with a second
MPI installed is not the one just built — the same two-MPI confusion the wheel
exists to prevent.

``nm -D`` and ``readelf -d`` supply the facts; everything here is the parsing,
kept apart from the MPICH policy in :mod:`wheelbuild.mpich` so it can be
tested against recorded output rather than a built library.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from wheelbuild._process import capture

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

#: One line of ``nm`` output: an optional address, a one-letter type, a name.
#: Undefined symbols have no address, which is why the address is optional.
_SYMBOL_LINE = re.compile(
    r"^\s*(?P<address>[0-9a-fA-F]+)?\s+(?P<type>[A-Za-z?])\s+(?P<name>\S+)\s*$"
)

#: ``nm`` type letters meaning the library references a symbol it does not
#: define. ``U`` is a plain undefined reference; the lowercase ``v`` and ``w``
#: are undefined weak ones (their uppercase counterparts are *defined* weak
#: symbols, which the library does provide).
UNDEFINED_TYPES = frozenset({"U", "v", "w"})

#: A ``readelf -d`` entry, whose value is bracketed:
#: ``0x…e (SONAME) Library soname: [libmpi.so.12]``. Not every tag has a
#: bracketed value — ``(FLAGS) SYMBOLIC`` does not — so the gap before the
#: bracket must stop at the end of the line. Allowed to cross one, a tag with
#: no value reaches down and claims the next entry's, which both mislabels
#: that value and consumes the entry it came from.
_DYNAMIC_ENTRY = re.compile(r"\((?P<tag>[A-Z_]+)\)[^\[\n]*\[(?P<value>[^\]]*)\]")

#: Prefixes of the symbols an MPI library is expected to supply: the standard
#: ``MPI_``/``PMPI_`` entry points, MPICH's internal ``MPIR_``/``MPII_``/
#: ``MPID_``/``MPIU_`` layers and its ``MPIX_`` extensions, and ``MPL``, the
#: MPICH portability library linked into ``libmpi``. Everything else a Fortran
#: binding layer references — libc, libgfortran, the loader — comes from
#: somewhere else entirely and is auditwheel's business, not libmpi's.
MPI_SYMBOL_PREFIXES = ("MPI", "PMPI", "MPL")


def _names(text: str, *, undefined: bool) -> set[str]:
    """Return the symbol names in ``nm`` output on one side of the divide."""
    found = set()
    for line in text.splitlines():
        match = _SYMBOL_LINE.match(line)
        if match is None:
            continue
        if (match["type"] in UNDEFINED_TYPES) is undefined:
            # A versioned reference such as ``memcpy@GLIBC_2.14`` names the
            # same symbol as a plain one; the version is the loader's concern.
            found.add(match["name"].partition("@")[0])
    return found


def undefined_symbols(text: str) -> set[str]:
    """Return the symbols ``nm -D`` output shows as referenced but not defined.

    Args:
        text: Output of ``nm -D`` over a shared library.

    Returns:
        The names, with any ``@version`` suffix removed.
    """
    return _names(text, undefined=True)


def defined_symbols(text: str) -> set[str]:
    """Return the symbols ``nm -D`` output shows the library providing.

    Args:
        text: Output of ``nm -D`` over a shared library.

    Returns:
        The names, with any ``@version`` suffix removed.
    """
    return _names(text, undefined=False)


def mpi_symbols(names: Iterable[str]) -> set[str]:
    """Narrow a set of symbols to the ones an MPI library is meant to supply.

    Args:
        names: Symbol names, as the parsers above report them.

    Returns:
        Those beginning with one of :data:`MPI_SYMBOL_PREFIXES`.
    """
    return {name for name in names if name.startswith(MPI_SYMBOL_PREFIXES)}


def parse_soname(text: str) -> str | None:
    """Return the ``DT_SONAME`` a ``readelf -d`` dump records.

    Args:
        text: Output of ``readelf -d`` over a shared library.

    Returns:
        The soname, or ``None`` when the library declares none.
    """
    for match in _DYNAMIC_ENTRY.finditer(text):
        if match["tag"] == "SONAME":
            return match["value"]
    return None


def parse_needed(text: str) -> frozenset[str]:
    """Return the ``DT_NEEDED`` libraries a ``readelf -d`` dump records.

    Args:
        text: Output of ``readelf -d`` over a shared library.

    Returns:
        The sonames the library asks the loader for.
    """
    return frozenset(
        match["value"]
        for match in _DYNAMIC_ENTRY.finditer(text)
        if match["tag"] == "NEEDED"
    )


def parse_runpath(text: str) -> tuple[str, ...]:
    """Return the library search path a ``readelf -d`` dump records.

    A shared object carries its search path as ``DT_RUNPATH``, or as the older
    ``DT_RPATH`` when it was linked before either the linker or patchelf
    started preferring the former. Both spell the same thing — a
    colon-separated list of directories, each of which may be relative to the
    loading object through ``$ORIGIN`` — so both are read here.

    Args:
        text: Output of ``readelf -d`` over a shared library.

    Returns:
        The directories, in the order the loader searches them, empty when the
        library records no search path at all.
    """
    directories: list[str] = []
    for match in _DYNAMIC_ENTRY.finditer(text):
        if match["tag"] in {"RUNPATH", "RPATH"}:
            directories.extend(entry for entry in match["value"].split(":") if entry)
    return tuple(directories)


def read_symbols(library: Path) -> tuple[set[str], set[str]]:
    """Read a shared library's dynamic symbol table.

    Args:
        library: Path of the shared library to inspect.

    Returns:
        Its defined symbols and its undefined ones, in that order.

    Raises:
        subprocess.CalledProcessError: When ``nm`` cannot read the file.
    """
    text = capture(["nm", "--dynamic", str(library)])
    return defined_symbols(text), undefined_symbols(text)


def read_dynamic(library: Path) -> tuple[str | None, frozenset[str]]:
    """Read a shared library's soname and the libraries it needs.

    Args:
        library: Path of the shared library to inspect.

    Returns:
        Its ``DT_SONAME`` and its ``DT_NEEDED`` entries, in that order.

    Raises:
        subprocess.CalledProcessError: When ``readelf`` cannot read the file.
    """
    text = capture(["readelf", "--dynamic", str(library)])
    return parse_soname(text), parse_needed(text)


def read_runpath(library: Path) -> tuple[str, ...]:
    """Read the search path a shared library asks the loader to use.

    Args:
        library: Path of the shared library to inspect.

    Returns:
        Its ``DT_RUNPATH`` (or ``DT_RPATH``) directories, in search order.

    Raises:
        subprocess.CalledProcessError: When ``readelf`` cannot read the file.
    """
    return parse_runpath(capture(["readelf", "--dynamic", str(library)]))


#: A register operand in ``objdump`` disassembly that only exists on a machine
#: newer than the x86-64 baseline: ``%ymm0`` needs AVX, ``%zmm0`` AVX-512.
#: Plain SSE2 code — everything a compiler emits for generic x86-64 — uses
#: ``%xmm`` and never these.
_WIDE_VECTOR_REGISTER = re.compile(r"%(?P<kind>[yz]mm)\d+")


def wide_vector_registers(disassembly: str) -> set[str]:
    """Return the post-baseline vector register kinds a disassembly uses.

    A manylinux wheel has to run on every x86-64 machine, not the one that
    compiled it. The usual way that breaks is a build system whose default is
    ``-march=native``: the result runs on the builder and dies with an illegal
    instruction on an older CPU, which is why :mod:`wheelbuild.petsc` pins
    OpenBLAS to dynamic dispatch. A library compiled for the baseline uses
    only the SSE2 ``%xmm`` registers, so finding ``%ymm`` or ``%zmm`` in one is
    the evidence that a native-tuning flag survived.

    Args:
        disassembly: Output of ``objdump --disassemble`` over a library.

    Returns:
        The register kinds found, such as ``{"ymm"}``. Empty for a baseline
        build.
    """
    return {match["kind"] for match in _WIDE_VECTOR_REGISTER.finditer(disassembly)}


def read_wide_vector_registers(library: Path) -> set[str]:
    """Read whether a library was compiled for a newer CPU than the baseline.

    Args:
        library: Path of the shared library to inspect.

    Returns:
        The post-baseline vector register kinds it uses; see
        :func:`wide_vector_registers`.

    Raises:
        subprocess.CalledProcessError: When ``objdump`` cannot read the file.
    """
    return wide_vector_registers(
        capture(["objdump", "--disassemble", "--no-show-raw-insn", str(library)])
    )
