"""Put the published release to an index, as a user reaches it.

Everything else in this suite proves a wheel that is a file on disk. What no
local run can prove is the half a release adds: that the three distributions
resolve each other by name from an index. Three things about that are decided
by the upload rather than by the build, and each of them is a case here.

* **``pip install dolfinx-solver`` brings a working DOLFINx.** The
  meta-package has no binaries and one pinned dependency (spec §1), so the
  install is the resolver following that pin to a variant wheel, on a machine
  the build never touched. The same numerical smoke test the wheel suite runs
  is what says the result computes.
* **The refusals name the distribution pip recorded.** ``dolfinx_solver.
  _bootstrap`` reads which wheel it is out of the installed distributions
  rather than being told (ticket 23), and the meta-package is not one of
  them. A stray ``petsc4py`` put here has to be refused with
  ``dolfinx-solver-complex`` in the message — the variant the meta-package
  resolved to — and not the bare name a user typed. It is fabricated rather
  than installed for the same reason the wheel suite fabricates it: PyPI's
  ``petsc4py`` is source-only, so installing it really would compile a whole
  second PETSc before the guard ever ran. What is real here is the
  distribution on the other side of the refusal, which is the half that
  needed an index.
* **The two variants refuse to share an environment.** They are one payload
  at the same paths compiled against a different ``PetscScalar`` (ticket 34),
  and until both are installable by name there is nowhere that case can be
  built at all: each CI job downloads its own variant's artefact and has only
  one wheel.

The first case runs in its own venv and the last in another, because
installing the second variant overwrites the first one's files: the
environment it builds is unusable by construction, which is the point of it.
The two installs there are sequential rather than one ``pip install A B``,
because sequential is the shape a user reaches it by — a working install,
then a second one on top of it.

Nothing is installed until the named index has been shown to serve this
release *on its own*. pip treats ``--index-url`` and ``--extra-index-url`` as
one namespace, and a dry run needs the second for its dependencies, so a
plain install could quietly resolve our own distributions from PyPI and
report success about a release TestPyPI never received.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from wheelbuild import assemble, publish
from wheelbuild._process import capture, check_call
from wheelbuild.version import __version__
from wheeltest import environment, foreign

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

#: What a release is, from the module that gathers one: the bare name a user
#: installs, the variants it can resolve to, and the indexes both ends of the
#: release path talk about. Imported rather than restated — a distribution
#: renamed in one of the two would otherwise publish under one name and be
#: checked under another.
META_DISTRIBUTION = publish.META_DISTRIBUTION
VARIANT_DISTRIBUTIONS = publish.VARIANT_DISTRIBUTIONS
INDEXES = publish.INDEXES

#: The variant the meta-package pins, which is the one the payload's messages
#: have to name once pip has resolved it (ticket 23). Read out of
#: ``meta/pyproject.toml`` at run time, because that dependency line is the
#: decision itself rather than a copy of one.
PINNED_DISTRIBUTION = publish.pinned_distribution()

#: The scalar type that variant carries, which is what the smoke stage has to
#: be told to expect. Taken off the pinned distribution's name rather than
#: from the build drivers' default, so one flip of the pin moves both.
PINNED_SCALAR_TYPE = PINNED_DISTRIBUTION.removeprefix(f"{META_DISTRIBUTION}-")

#: The stages this check runs against the installed release, with what each
#: needs beyond ``--site``. They are the suite's own two cheapest — the
#: import check the build itself runs (ticket 04) and the numerical smoke
#: test — rather than a second definition of what a working install is. The
#: variant is named rather than read from the environment: what is under test
#: is the one the meta-package's pin resolves to.
STAGES = (
    ("wheelbuild.import_check", ("--dolfinx",)),
    ("wheeltest.smoke", ("--scalar-type", PINNED_SCALAR_TYPE)),
)

#: How long to keep asking the index for the release. An upload is accepted
#: before it is installable, and the gap is seconds to a minute — long enough
#: that a check running straight after the upload fails a release that is
#: perfectly fine.
INDEX_ATTEMPTS = 6
INDEX_DELAY_SECONDS = 20.0


def index_arguments(index: str) -> list[str]:
    """Return the pip arguments that point an install at one index.

    Args:
        index: ``pypi`` or ``testpypi``.

    Returns:
        The arguments, empty for the default index.

    Raises:
        ValueError: When the index is not one this project publishes to.
    """
    if index not in INDEXES:
        raise ValueError(
            f"there is no {index!r} index; this project publishes to "
            f"{', '.join(sorted(INDEXES))}"
        )
    simple = INDEXES[index].simple
    if simple is None:
        return []
    return ["--index-url", simple, "--extra-index-url", publish.FALLBACK_SIMPLE_URL]


def strict_index_arguments(index: str) -> list[str]:
    """Return the pip arguments that reach one index and nothing else.

    Args:
        index: ``pypi`` or ``testpypi``.

    Returns:
        The arguments, empty for the default index.

    Raises:
        ValueError: When the index is not one this project publishes to.
    """
    if index not in INDEXES:
        raise ValueError(
            f"there is no {index!r} index; this project publishes to "
            f"{', '.join(sorted(INDEXES))}"
        )
    simple = INDEXES[index].simple
    return [] if simple is None else ["--index-url", simple]


def wait_for_release(
    dest: Path,
    *,
    index: str = "pypi",
    version: str = __version__,
    distributions: Sequence[str] = publish.DISTRIBUTIONS,
    python: str = sys.executable,
    attempts: int = INDEX_ATTEMPTS,
    delay: float = INDEX_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until the named index serves this release, and prove it did.

    Downloads every published distribution from that index with no fallback
    and no dependency resolution. It is both halves of one problem: an upload
    is not installable the instant it is accepted, and an install that is
    allowed a second index cannot say which one answered it.

    Args:
        dest: Directory to download into. Nothing is installed from it; what
            matters is that the download succeeded.
        index: Which index the release was published to.
        version: The version that was published.
        distributions: What a release publishes.
        python: Interpreter whose pip does the asking. The driver's own —
            no environment has been made yet.
        attempts: How many times to ask before giving up.
        delay: Seconds between attempts.
        sleep: How to wait. Injected so the retry can be tested without one.

    Raises:
        subprocess.CalledProcessError: When the index still does not serve
            the release on the last attempt.
    """
    command = [
        python,
        "-m",
        "pip",
        "download",
        "--no-deps",
        "--dest",
        str(dest),
        *strict_index_arguments(index),
        *requirements(distributions, version),
    ]
    for attempt in range(1, attempts):
        try:
            check_call(command)
        except subprocess.CalledProcessError:
            print(
                f"{index} does not serve {version} yet (attempt {attempt} of "
                f"{attempts}); waiting {delay:.0f}s",
                flush=True,
            )
            sleep(delay)
        else:
            return
    # The last attempt is outside the loop: its failure is the caller's, and
    # a check reporting "not published yet" after two minutes of trying would
    # be hiding whatever pip actually said.
    check_call(command)


def requirements(names: Iterable[str], version: str = __version__) -> list[str]:
    """Return the exact requirements that name this release.

    Args:
        names: The distributions to install.
        version: The version they were published under.

    Returns:
        One ``name==version`` per distribution. Exact, because the index may
        already hold a later release and this check is about the one that was
        just uploaded.
    """
    return [f"{name}=={version}" for name in names]


def install(
    venv_dir: Path,
    install_requirements: Sequence[str],
    *,
    base_python: str = sys.executable,
    index: str = "pypi",
) -> Path:
    """Make a clean venv and install distributions into it from an index.

    No retry: :func:`wait_for_release` has already shown that the index
    serves this release, so a failure here is a failure of the install rather
    than of the upload's timing.

    Args:
        venv_dir: Directory for the environment, replaced if it exists — a
            reused one can still hold the previous run's install, which is
            how a check passes against a release that is no longer there.
        install_requirements: What to install, as pip takes it.
        base_python: Interpreter the venv is made from.
        index: Which index to install from. Dependencies may come from
            elsewhere, which is what the fallback is for.

    Returns:
        The environment's interpreter.

    Raises:
        subprocess.CalledProcessError: When the install fails.
    """
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    check_call([base_python, "-m", "venv", str(venv_dir)])
    python = environment.venv_python(venv_dir)
    check_call([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    add(python, install_requirements, index=index)
    return python


def add(
    python: Path, install_requirements: Sequence[str], *, index: str = "pypi"
) -> None:
    """Install more into an environment that already has some of this release.

    Args:
        python: The environment's interpreter.
        install_requirements: What to install, as pip takes it.
        index: Which index to install from.

    Raises:
        subprocess.CalledProcessError: When the install fails.
    """
    check_call(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            *index_arguments(index),
            *install_requirements,
            *assemble.CHECK_REQUIREMENTS,
        ]
    )


def installed_names(python: Path) -> list[str]:
    """Return the normalised names of what is installed in an environment.

    Args:
        python: The environment's interpreter.

    Returns:
        Every installed distribution, as pip spells it after normalisation.
    """
    frozen = capture([str(python), "-m", "pip", "list", "--format=freeze"])
    return [
        line.partition("==")[0].strip().lower().replace("_", "-")
        for line in frozen.splitlines()
        if "==" in line
    ]


def resolution_problem(
    names: Iterable[str],
    pinned: str = PINNED_DISTRIBUTION,
    variants: Sequence[str] = VARIANT_DISTRIBUTIONS,
) -> str | None:
    """Report a meta-package install that did not bring the variant it pins.

    Args:
        names: What the environment holds, as :func:`installed_names` reports.
        pinned: The variant the meta-package depends on.
        variants: Every variant there is a distribution for.

    Returns:
        A message, or ``None`` when exactly the pinned variant is installed.
    """
    installed = sorted(set(variants).intersection(names))
    if installed == [pinned]:
        return None
    if not installed:
        return (
            f"installing {META_DISTRIBUTION} brought no variant wheel: the "
            f"environment holds none of {', '.join(variants)}. The "
            "meta-package has no binaries of its own, so an install of it "
            "that resolved nothing is an install with no DOLFINx in it "
            "(spec §1)."
        )
    return (
        f"installing {META_DISTRIBUTION} brought {', '.join(installed)}, and "
        f"it pins {pinned}. What `pip install {META_DISTRIBUTION}` means is "
        "the one decision the meta-package exists to make."
    )


def import_problem(
    python: Path,
    *,
    refused: bool,
    expected: str = "",
    why: str = "",
    path: Path | None = None,
) -> str | None:
    """Put ``import dolfinx_solver`` to an installed environment.

    The judgement is :mod:`wheeltest.foreign`'s, which is where what a
    refusal has to look like is written down; what is different here is only
    that the environment was installed from an index by name rather than
    fabricated beside a wheel.

    Args:
        python: The environment's interpreter.
        refused: Whether the import has to fail.
        expected: A phrase the refusal has to contain.
        why: What a wrong answer would mean for a user.
        path: A directory to put ahead of the installed packages.

    Returns:
        A message, or ``None`` when the environment answered as it must.
    """
    case = foreign.Case(
        name=f"the environment installed in {python.parent.parent}",
        refused=refused,
        expected=expected,
        why=why,
    )
    return foreign.put_case(case, path, str(python))


def check_meta(python: Path, work_dir: Path) -> str | None:
    """Prove the environment ``pip install dolfinx-solver`` produces.

    Args:
        python: Interpreter of the environment it was installed into.
        work_dir: Scratch directory for the fabricated stray distribution.

    Returns:
        The first problem, or ``None`` when every check passed.
    """
    problem = resolution_problem(installed_names(python))
    if problem is not None:
        return problem

    problem = import_problem(
        python,
        refused=False,
        why=(
            "this is the environment `pip install dolfinx-solver` produces, "
            "and it is the first thing a user does"
        ),
    )
    if problem is not None:
        return problem

    site = environment.site_packages(python)
    child = environment.child_environment()
    for module, arguments in STAGES:
        try:
            check_call(
                [str(python), "-m", module, "--site", str(site), *arguments], env=child
            )
        except subprocess.CalledProcessError:
            return (
                f"the release installed from the index failed {module}. Its "
                "own message is above; what is different about this "
                "environment is that nothing in it came from a build — the "
                "wheel, the mpich it loads and the upstream trio were all "
                "resolved by name."
            )

    return import_problem(
        python,
        refused=True,
        expected=PINNED_DISTRIBUTION,
        path=foreign.write_installed_petsc4py(work_dir / "stray-petsc4py"),
        why=(
            "the payload reads which wheel it is from the installed "
            "distributions (ticket 23). Installed by name, that is whatever "
            f"the resolver chose — {PINNED_DISTRIBUTION} — and a message "
            f"naming {META_DISTRIBUTION} instead would tell a user to "
            "reinstall a package that ships no files"
        ),
    )


def check_both_variants(python: Path) -> str | None:
    """Prove an environment holding both variants is refused.

    Args:
        python: Interpreter of the environment both were installed into.

    Returns:
        A message, or ``None`` when the import refused it naming both.
    """
    for distribution in VARIANT_DISTRIBUTIONS:
        problem = import_problem(
            python,
            refused=True,
            expected=distribution,
            why=(
                "both variants ship the same payload at the same paths with a "
                "different PetscScalar compiled in, so one wheel's files have "
                "overwritten the other's and which scalar type the process "
                "gets is decided by the order pip unpacked them (ticket 34). "
                "The refusal has to name both, because uninstalling either "
                "one alone removes files the other's RECORD claims"
            ),
        )
        if problem is not None:
            return problem
    return None


def check(
    work_dir: Path,
    *,
    base_python: str = sys.executable,
    index: str = "pypi",
    version: str = __version__,
) -> str | None:
    """Install this release from an index and put every case to it.

    Args:
        work_dir: Scratch directory for the environments.
        base_python: Interpreter the venvs are made from.
        index: Which index the release was published to.
        version: The version that was published.

    Returns:
        The first problem, or ``None`` when the release answered every case.
    """
    wait_for_release(
        work_dir / "served", index=index, version=version, python=base_python
    )

    meta_python = install(
        work_dir / "meta-venv",
        requirements([META_DISTRIBUTION], version),
        base_python=base_python,
        index=index,
    )
    problem = check_meta(meta_python, work_dir)
    if problem is not None:
        return problem

    # One variant, then the other on top of it: the shape a user reaches this
    # by is a working install and a second one over it, and the first install
    # succeeding is part of what makes the refusal meaningful.
    first, *rest = VARIANT_DISTRIBUTIONS
    both_python = install(
        work_dir / "both-venv",
        requirements([first], version),
        base_python=base_python,
        index=index,
    )
    add(both_python, requirements(rest, version), index=index)
    return check_both_variants(both_python)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the post-publish check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--index",
        choices=sorted(INDEXES),
        default="pypi",
        help="the index the release was published to",
    )
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    problem = check(args.work_dir, index=args.index)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"{META_DISTRIBUTION}=={__version__} installed from {args.index}, "
        f"resolved to {PINNED_DISTRIBUTION}, solved the smoke problem, "
        "refused a stray petsc4py naming the variant the resolver chose, and "
        f"refused {' and '.join(VARIANT_DISTRIBUTIONS)} installed together"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
