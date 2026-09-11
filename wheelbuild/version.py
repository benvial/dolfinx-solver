"""The packaged version, read without importing the package.

``dolfinx_solver/__init__.py`` loads ``mpi4py.MPI`` on import — that ordering
is the whole point of the package — so importing anything from it, submodule
included, needs a working MPI stack. The build tooling runs before there is
one: the tag check fires on a tag push, the assembly steps run in a container
where nothing is installed yet. They read the literals straight out of the
version module's source file instead, the way setuptools does.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

#: The version module inside the package, which imports nothing itself.
VERSION_SOURCE = (
    Path(__file__).resolve().parent.parent / "dolfinx_solver" / "_version.py"
)


def load(source: Path = VERSION_SOURCE) -> ModuleType:
    """Execute the version module on its own, outside its package.

    Args:
        source: Path of the version module's source file.

    Returns:
        The loaded module, carrying ``__version__``, ``DOLFINX_VERSION`` and
        ``RELEASE``.

    Raises:
        ImportError: When the file cannot be loaded as a module.
    """
    spec = importlib.util.spec_from_file_location("_dolfinx_solver_version", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the version module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VERSION = load()

#: This wheel's version, from ``dolfinx_solver.__version__``.
__version__: str = _VERSION.__version__

#: Upstream DOLFINx release this wheel ships, as upstream tags it.
DOLFINX_VERSION: str = _VERSION.DOLFINX_VERSION

#: That release without any packaging segment.
RELEASE: str = _VERSION.RELEASE
