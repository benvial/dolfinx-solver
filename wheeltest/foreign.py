"""Build the environments the wheel refuses, and check that it refuses them.

``dolfinx_solver._bootstrap`` turns two environments away at import: one
where a PyPI ``petsc``, ``petsc4py``, ``slepc`` or ``slepc4py`` distribution
is installed, and one where some other petsc4py would be imported ahead of
the vendored copy. Its unit tests already prove the judgements; what they
cannot prove is that the guard is wired into the installed package's import
path at all, because they call the functions directly.

So this stage constructs both environments against the wheel as installed,
in child interpreters, and asserts three things:

* the shadowing case is refused, naming the file that would have been
  imported,
* the installed-distribution case is refused, naming what to uninstall,
* and the clean environment is *not* refused — which is the half that a
  guard written too broadly would fail, and the half a user notices.

Each case is a child process because ``import dolfinx_solver`` happens once
per interpreter and loads MPI when it succeeds. Running them in this one
would prove only whichever came first.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

#: What a child runs: the import the guard is attached to, and nothing else.
IMPORT_PROGRAM = "import dolfinx_solver"

#: Keeps the working directory off the child's module path. The suite is
#: normally run from a checkout, which holds the source of the very package
#: being imported; without this the child would import that instead of the
#: installed one and refuse the environment for the wrong reason. The
#: fabricated ``PYTHONPATH`` each case sets is unaffected, which is what
#: makes this safe to apply to all three.
SAFE_PATH = ("-P",)

#: The foreign distribution the installed-distribution case pretends is
#: there. Its name is one of the four ``_bootstrap`` refuses; its version is
#: one no real petsc4py has, so a run that somehow met the real project
#: still reads as this test's doing.
FOREIGN_DISTRIBUTION = "petsc4py"
FOREIGN_VERSION = "3.99.0"


class Case(NamedTuple):
    """One environment put to the installed wheel.

    Attributes:
        name: What it is called in the report.
        refused: Whether ``import dolfinx_solver`` has to fail in it.
        expected: A phrase the refusal has to contain, so a failure for an
            unrelated reason is not read as the guard working.
        why: What a wrong answer here would mean for a user.
    """

    name: str
    refused: bool
    expected: str
    why: str


SHADOWING = Case(
    name="a petsc4py earlier on sys.path",
    refused=True,
    expected="is not the build shipped inside this wheel",
    why=(
        "a second PETSc would be imported into the process beside the "
        "vendored one, with a different scalar type and a different libmpi "
        "behind it. There is no error that follows from that — the symptoms "
        "are a hang or a wrong answer — so the import is where it has to be "
        "caught (spec §1)."
    ),
)

INSTALLED = Case(
    name="a petsc4py distribution installed beside ours",
    refused=True,
    expected="Uninstall",
    why=(
        "pip installing the PyPI project writes its files exactly where the "
        "vendored ones live, so the import path looks untouched and only the "
        "installed distribution gives it away. A user who did this needs to "
        "be told which package to remove and that reinstalling restores the "
        "files it overwrote."
    ),
)

CLEAN = Case(
    name="the environment the wheel installs into",
    refused=False,
    expected="",
    why=(
        "a guard that fires here fires on every correct install. This is the "
        "half of the check that the other two cannot show."
    ),
)

#: The three environments, in the order the report lists them.
CASES = (SHADOWING, INSTALLED, CLEAN)


def write_shadowing_petsc4py(root: Path) -> Path:
    """Put a petsc4py that is not ours where an import would find it first.

    Args:
        root: Directory to build it in.

    Returns:
        The directory to put on ``PYTHONPATH``.
    """
    package = root / "petsc4py"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        "# A petsc4py that is not the vendored one, for wheeltest.foreign.\n",
        encoding="utf-8",
    )
    return root


def write_installed_petsc4py(root: Path) -> Path:
    """Put the installed metadata of a foreign petsc4py where pip would.

    The distribution is what this case is about, so it is metadata with no
    package beside it: that is also the arrangement that tells the two
    refusals apart, since nothing here is importable.

    Args:
        root: Directory to build it in.

    Returns:
        The directory to put on ``PYTHONPATH``.
    """
    dist_info = root / f"{FOREIGN_DISTRIBUTION}-{FOREIGN_VERSION}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: {FOREIGN_DISTRIBUTION}\n"
        f"Version: {FOREIGN_VERSION}\n",
        encoding="utf-8",
    )
    return root


#: How each refused case is built, keyed by the case.
BUILDERS = {
    SHADOWING.name: write_shadowing_petsc4py,
    INSTALLED.name: write_installed_petsc4py,
}


def outcome_problem(case: Case, returncode: int, output: str) -> str | None:
    """Report a case the installed wheel answered wrongly.

    Args:
        case: The environment that was put to it.
        returncode: What the child interpreter exited with.
        output: What it wrote to its standard error.

    Returns:
        A message, or ``None`` when the wheel answered as it must.
    """
    if case.refused and returncode == 0:
        return (
            f"{IMPORT_PROGRAM} succeeded with {case.name}. It has to fail "
            f"there: {case.why}"
        )
    if not case.refused and returncode != 0:
        return (
            f"{IMPORT_PROGRAM} failed in {case.name}:\n{output.strip()}\n"
            f"That is the environment every user installs into, and {case.why}"
        )
    if case.refused and case.expected not in output:
        return (
            f"{IMPORT_PROGRAM} failed with {case.name}, but not with the "
            f"message the guard raises — nothing in its output contains "
            f"{case.expected!r}:\n{output.strip()}\nAn import that fails for "
            "some other reason is not the check passing."
        )
    return None


def child_environment(path: Path | None) -> dict[str, str]:
    """Return the environment a case's child interpreter runs in.

    Args:
        path: Directory to put ahead of the installed packages, or ``None``
            for the clean case.

    Returns:
        The environment.
    """
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if path is not None:
        environment["PYTHONPATH"] = str(path)
    return environment


def run_case(case: Case, root: Path, python: str = sys.executable) -> str | None:
    """Put one environment to the installed wheel.

    Args:
        case: The environment to build and the answer required.
        root: Directory to build it in.
        python: The interpreter the wheel is installed for.

    Returns:
        A message, or ``None`` when the wheel answered as it must.
    """
    builder = BUILDERS.get(case.name)
    path = None if builder is None else builder(root / case.name.replace(" ", "-"))
    completed = subprocess.run(
        [python, *SAFE_PATH, "-c", IMPORT_PROGRAM],
        env=child_environment(path),
        capture_output=True,
        text=True,
        check=False,
    )
    return outcome_problem(case, completed.returncode, completed.stderr)


def check(site: Path, root: Path, python: str = sys.executable) -> str | None:
    """Put every environment to the installed wheel, and report the first wrong.

    Args:
        site: Where the payload is installed. Reported, so a failure says
            which install was being asked.
        root: Directory to build the environments in.
        python: The interpreter the wheel is installed for.

    Returns:
        A message, or ``None`` when all three are answered correctly.
    """
    for case in CASES:
        problem = run_case(case, root, python)
        if problem is not None:
            return f"{problem} (the wheel under test is installed in {site})"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the foreign-petsc4py check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="where the fabricated environments are built; a temporary "
        "directory by default",
    )
    args = parser.parse_args(argv)

    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        problem = check(args.site, args.work_dir)
    else:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="wheeltest-foreign-") as temporary:
            problem = check(args.site, Path(temporary))

    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"the installed wheel refused {sum(case.refused for case in CASES)} "
        "foreign PETSc environments and imported cleanly in the one a user "
        "gets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
