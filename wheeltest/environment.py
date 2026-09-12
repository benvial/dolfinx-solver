"""The wheelhouse, the clean venv, and the environment the stages run in.

Every stage of the suite needs the same three things, and each of them is a
decision rather than plumbing:

* **Which wheel.** A wheelhouse is a directory, and there are two of them in
  a finished build: ``$build_root/wheelhouse`` holds the published artefact,
  ``$prefix/wheelhouse`` holds the petsc4py, slepc4py and fenics_dolfinx
  wheels the earlier stages built for themselves and that must never be
  published (spec §6). Globbing the distribution name keeps the suite off
  the second one, and finding two wheels is refused rather than guessed at.
* **A venv with nothing else in it.** The wheel's dependency metadata is
  half the product: it is what brings the PyPI ``mpich`` wheel, mpi4py and
  the upstream trio along with the install (spec §5, §10). An environment
  that already had them would prove nothing about the metadata, so the venv
  is made fresh and the dependencies are resolved from an index — or from a
  local wheelhouse when ``--offline`` says there is none.
* **No ``LD_LIBRARY_PATH``.** The whole point of the vendored layout is that
  the wheel's libraries find each other through the rpaths they carry. A
  search path inherited from the caller would let them find each other some
  other way, and the suite would pass for a wheel that works on this machine
  alone.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from wheelbuild import assemble
from wheelbuild._process import check_call

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Repository root. The stages run ``python -m wheelbuild...`` and
#: ``python -m wheeltest...`` inside the venv, and neither package is
#: installed into anything, so this goes on their ``PYTHONPATH``.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: What our wheel is called in a wheelhouse that may hold others.
WHEEL_GLOB = f"{assemble.DISTRIBUTION}-*.whl"

#: Where a warm container build leaves the wheel, relative to the repository.
#: The host half of ``$build_root/wheelhouse``, which is what
#: ``scripts/build-in-container.sh`` mounts the cache at.
DEFAULT_WHEELHOUSE = Path(".build-cache") / "wheelhouse"

#: The process launcher the interop stage and the parallel demos run under.
#: It comes from the PyPI ``mpich`` wheel installed beside the interpreter —
#: hydra, matching the ``libmpi`` mpi4py loads — and never from the machine,
#: where an Open MPI ``mpiexec`` would start two one-rank jobs that each
#: think they are alone (spec §11).
LAUNCHER = "mpiexec"


def find_wheel(wheelhouse: Path) -> Path:
    """Return the one publishable wheel in a wheelhouse.

    Args:
        wheelhouse: Directory to look in.

    Returns:
        The wheel.

    Raises:
        FileNotFoundError: When the directory holds no wheel of ours.
        ValueError: When it holds more than one, which is a stale build
            beside a fresh one and picking either silently is how a suite
            comes to pass against a wheel nobody meant to test.
    """
    wheels = sorted(wheelhouse.glob(WHEEL_GLOB))
    if not wheels:
        raise FileNotFoundError(
            f"no {WHEEL_GLOB} in {wheelhouse}. The container build leaves the "
            "publishable wheel in $build_root/wheelhouse, which is not the "
            "$prefix/wheelhouse holding the petsc4py, slepc4py and "
            "fenics_dolfinx wheels the earlier stages built for themselves."
        )
    if len(wheels) > 1:
        raise ValueError(
            f"{wheelhouse} holds {len(wheels)} wheels: "
            f"{', '.join(wheel.name for wheel in wheels)}. Which one the "
            "suite should prove is not something to guess at; remove the "
            "ones that are not being released."
        )
    return wheels[0]


def venv_python(venv_dir: Path) -> Path:
    """Return the interpreter inside a venv.

    Args:
        venv_dir: The environment's directory.

    Returns:
        Its ``python``.
    """
    return venv_dir / "bin" / "python"


def launcher(python: Path) -> Path:
    """Return the MPI process launcher installed beside an interpreter.

    Args:
        python: Interpreter of the environment the wheel is installed in.

    Returns:
        The ``mpiexec`` the PyPI ``mpich`` wheel put there.

    Raises:
        FileNotFoundError: When it is not there. That means the install did
            not bring the mpich wheel, so the parallel stages would either
            not run or run under whatever ``mpiexec`` the machine has, which
            is the wrong MPI and starts ranks that cannot see each other.
    """
    found = python.parent / LAUNCHER
    if not found.exists():
        raise FileNotFoundError(
            f"no {LAUNCHER} at {found}. It comes from the PyPI mpich wheel "
            "the wheel's own metadata depends on, so a venv without it is a "
            "wheel whose dependencies did not install — and the machine's "
            "own mpiexec is not a substitute, since a launcher from another "
            "MPI starts ranks that never form one communicator."
        )
    return found


def install_arguments(
    wheel: Path,
    requirements: Sequence[str] = (),
    *,
    python: Path,
    wheelhouse: Path | None = None,
) -> list[str]:
    """Return the command that installs the wheel into the clean venv.

    Args:
        wheel: The wheel to install.
        requirements: Anything else the venv needs, installed in the same
            command as the wheel rather than before it, so pip resolves them
            against the wheel's own dependencies.
        python: Interpreter of the venv.
        wheelhouse: Directory to resolve the dependencies from instead of an
            index. Given for an offline run, where the wheelhouse has to hold
            mpich, mpi4py, numpy, cffi and the upstream trio as well as ours,
            and ``packaging`` for the stages themselves.

    Returns:
        The argument vector.
    """
    command = [
        *assemble.install_arguments(wheel, python=str(python)),
        *requirements,
    ]
    if wheelhouse is not None:
        command += ["--no-index", "--find-links", str(wheelhouse)]
    return command


def child_environment(
    environment: Mapping[str, str] | None = None, repo_root: Path = REPO_ROOT
) -> dict[str, str]:
    """Return the environment a stage runs in.

    Args:
        environment: Environment to derive it from. Defaults to this
            process's.
        repo_root: Repository root, which holds the two driver packages the
            stages are run as modules of.

    Returns:
        The environment, with the caller's library search path gone and the
        repository on the module path.
    """
    child = assemble.audit_environment(environment)
    child["PYTHONPATH"] = str(repo_root)
    # An abi3 wheel installs on three interpreters, and the venv is thrown
    # away after the run; bytecode in it is at best noise in the diff of what
    # the suite left behind.
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    return child


def create(
    venv_dir: Path,
    wheel: Path,
    *,
    base_python: str = sys.executable,
    requirements: Sequence[str] = assemble.CHECK_REQUIREMENTS,
    wheelhouse: Path | None = None,
) -> Path:
    """Make a venv holding nothing but the wheel and what it brings.

    Args:
        venv_dir: Directory for the environment. Replaced if it exists: a
            reused one may still hold a previous wheel's files, which is how
            a suite passes against something that is no longer there.
        wheel: The wheel to install.
        base_python: Interpreter to make the environment from. This is the
            matrix axis — one abi3 wheel has to install and import on 3.12,
            3.13 and 3.14 (spec §4, §9).
        requirements: Anything else the venv needs. Defaults to what
            :mod:`wheelbuild.import_check` itself imports, which the wheel is
            not required to bring.
        wheelhouse: Directory to resolve dependencies from, for an offline
            run. It has to hold everything the wheel depends on plus
            ``requirements``; nothing is fetched, not even a pip upgrade.

    Returns:
        The environment's interpreter.
    """
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    check_call([base_python, "-m", "venv", str(venv_dir)])
    python = venv_python(venv_dir)
    if wheelhouse is None:
        # Skipped for an offline run, where it is the one command that would
        # reach the index anyway. The venv's own pip installs this wheel
        # fine; upgrading it is a convenience against an old bundled pip,
        # not something the install depends on.
        check_call([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    check_call(
        install_arguments(wheel, requirements, python=python, wheelhouse=wheelhouse)
    )
    return python


def site_packages(python: Path) -> Path:
    """Return where a venv's interpreter installs packages.

    Args:
        python: The interpreter.

    Returns:
        Its ``site-packages``, which is the directory the import check
        asserts every payload package resolved under.
    """
    return assemble.site_packages(str(python))
