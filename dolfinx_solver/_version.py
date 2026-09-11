"""The single source of the version (spec §3), kept importable on its own.

``pyproject.toml`` reads ``__version__`` from this module, and setuptools
parses the literal out of the source rather than importing it — which is why
these are plain string literals in a module that imports nothing: the package
``__init__`` loads MPI, and a build must not need a working MPI stack to learn
its own version number.
"""

from __future__ import annotations

#: The DOLFINx release these wheels build, with no packaging segment: the part
#: that has to match for this wheel to be a build of that DOLFINx.
RELEASE = "0.11.0"

#: Upstream DOLFINx release this wheel ships, spelled as upstream spells it —
#: including upstream's own ``.post`` segment, because the build scripts fetch
#: ``v`` plus this string as a git tag and ``v0.11.0`` is not a tag upstream
#: has.
DOLFINX_VERSION = "0.11.0.post0"

#: This wheel's version: :data:`DOLFINX_VERSION` with its ``.postN`` segment
#: moved forward for packaging-only fixes, so the first packaging fix on this
#: release is ``0.11.0.post1``. PEP 440 allows one post segment and upstream
#: already spent ``post0``, so the segment is shared rather than appended.
__version__ = "0.11.0.post0"
