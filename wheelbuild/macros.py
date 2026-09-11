"""Read the ``#define`` facts a configured build records about itself.

PETSc and SLEPc both write a configuration header into their install prefix —
``petscconf.h``, ``slepcconf.h`` — naming every feature the configure run
turned on and every ABI choice it baked in. That header is the only record of
what a multi-hour build actually produced, and it survives in the prefix long
after the configure log has scrolled away.

Reading it is how the drivers check their own work, for the reason
:mod:`wheelbuild.elf` gives: a fact read out of an installed file cannot be
shadowed by whatever else the machine has installed, while a fact obtained by
running a binary out of the prefix and asking it can be. The parsing lives
here, apart from the PETSc and SLEPc policy, so both can be tested against
recorded header text instead of a built library.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: An object-like macro: ``#define NAME`` with an optional value. Function-like
#: macros do not match, because the ``(`` follows the name with no space in
#: between and nothing here accepts that — which is what we want, since a
#: configuration header's facts are all object-like and a matched
#: ``#define PetscFoo(x)`` would report the wrong name.
_DEFINE = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+(?P<name>[A-Za-z_]\w*)"
    r"(?:[ \t]+(?P<value>[^\n]*?))?[ \t]*$",
    re.MULTILINE,
)


def defined_macros(header_text: str) -> dict[str, str]:
    """Return the object-like macros a header defines, by name.

    Args:
        header_text: Contents of a C header.

    Returns:
        Each macro's name mapped to its value, which is the empty string for a
        bare ``#define NAME``. A name defined more than once keeps the last
        value, as the preprocessor would.
    """
    return {
        match["name"]: match["value"] or "" for match in _DEFINE.finditer(header_text)
    }


def defined_names(header_text: str) -> frozenset[str]:
    """Return the names a header defines.

    Most of what a configuration header records is presence, not value:
    ``PETSC_HAVE_MUMPS`` is defined or it is not.

    Args:
        header_text: Contents of a C header.

    Returns:
        The macro names, values discarded.
    """
    return frozenset(defined_macros(header_text))


def read_defined_names(*headers: Path) -> frozenset[str]:
    """Return the names a set of installed headers define between them.

    Args:
        headers: Header files to read. All of them must exist; a build whose
            configuration header is missing has not finished installing, and
            treating that as "no features" would report the wrong problem.

    Returns:
        The union of their macro names.

    Raises:
        FileNotFoundError: When a header does not exist.
    """
    names: set[str] = set()
    for header in headers:
        names |= defined_names(header.read_text(encoding="utf-8"))
    return frozenset(names)
