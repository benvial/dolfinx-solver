"""Install the wheel once, then put every stage to it.

One entry point, because the expensive part of testing a wheel is not the
tests: it is the clean venv and the install that proves the metadata. The
stages are cheap next to it and they all want the same environment, so the
venv is made once here and each stage is run in it as a module.

Which stages run is a command-line choice for one reason — a failure is
usually chased by re-running the stage that failed, and making a whole venv
to do that is a minute of nothing. The default is all of them, in the order
below, which is cheapest-first: an install that did not bring the runtime
stack is reported by the import stage in seconds, before the demos have
compiled a form.

The stages are separate processes for a harder reason. ``import
dolfinx_solver`` loads MPI into the interpreter and cannot be undone, the
interop stage needs an interpreter started by ``mpiexec``, and the foreign
stage needs interpreters whose module path is wrong on purpose. One process
could host none of that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from wheelbuild._process import check_call
from wheeltest import environment

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class Stage(NamedTuple):
    """One thing the suite puts to the installed wheel.

    Attributes:
        name: What it is called on the command line.
        module: The module run in the venv, as ``python -m`` takes it.
        ranks: How many ranks it runs under.
        proves: What a pass means, for the run's own report.
    """

    name: str
    module: str
    ranks: int
    proves: str


#: The stages, cheapest first. The first is :mod:`wheelbuild.import_check`
#: itself: it is the same check the build runs against the staging site
#: (ticket 04), pointed at an installed venv's site-packages instead, and
#: writing a second one would be writing a second definition of what a
#: working import is.
STAGES = (
    Stage(
        name="imports",
        module="wheelbuild.import_check",
        ranks=1,
        proves=(
            "the payload imports from the installed site, in the order the "
            "linkage needs, with one MPI runtime in the process and complex "
            "scalars in PETSc"
        ),
    ),
    Stage(
        name="foreign",
        module="wheeltest.foreign",
        ranks=1,
        proves=(
            "the wheel refuses an environment where another petsc4py would "
            "be imported, and accepts the one a user installs into"
        ),
    ),
    Stage(
        name="smoke",
        module="wheeltest.smoke",
        ranks=1,
        proves=(
            "a complex Helmholtz problem solves onto its exact plane wave "
            "and SLEPc finds the Dirichlet Laplacian's first eigenvalues"
        ),
    ),
    Stage(
        name="interop",
        module="wheeltest.interop",
        ranks=2,
        proves=(
            "mpi4py, petsc4py and DOLFINx share one communicator across two "
            "ranks over a PT-SCOTCH-partitioned mesh"
        ),
    ),
    Stage(
        name="demos",
        module="wheeltest.demos",
        ranks=1,
        proves="upstream's own demo programs run against the wheel",
    ),
)

#: The stage whose sources come from outside the wheel, and which therefore
#: takes arguments none of the others do.
DEMOS_STAGE = "demos"

#: What the import-check stage needs beyond ``--site``: the staged-DOLFINx
#: half of the check, which for an installed wheel is always expected.
IMPORT_ARGUMENTS = ("--dolfinx",)


def stage_by_name(name: str) -> Stage:
    """Return a stage by its command-line name.

    Args:
        name: What the caller asked for.

    Returns:
        The stage.

    Raises:
        ValueError: When there is no such stage.
    """
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise ValueError(
        f"there is no {name!r} stage. The suite runs "
        f"{', '.join(stage.name for stage in STAGES)}."
    )


def command(
    stage: Stage,
    *,
    python: Path,
    site: Path,
    launcher: Path,
    work_dir: Path,
    demo_source: Path | None = None,
    demo_cache: Path | None = None,
) -> list[str]:
    """Return the command that runs one stage in the installed venv.

    Args:
        stage: The stage to run.
        python: The venv's interpreter.
        site: Its ``site-packages``, which every stage is told about so a
            failure can say which install it was asking.
        launcher: The ``mpiexec`` beside the interpreter.
        work_dir: Scratch directory for the stages that need one.
        demo_source: A DOLFINx source tree to take the demos from.
        demo_cache: Where to download the pinned release instead.

    Returns:
        The argument vector.
    """
    arguments = [str(python), "-m", stage.module, "--site", str(site)]
    if stage.module == STAGES[0].module:
        arguments += list(IMPORT_ARGUMENTS)
    if stage.name == DEMOS_STAGE:
        arguments += ["--work-dir", str(work_dir / stage.name)]
        if demo_source is not None:
            arguments += ["--source", str(demo_source)]
        elif demo_cache is not None:
            arguments += ["--cache", str(demo_cache)]
    if stage.ranks > 1:
        return [str(launcher), "-n", str(stage.ranks), *arguments]
    return arguments


def run(
    *,
    wheelhouse: Path,
    venv_dir: Path,
    work_dir: Path,
    stages: Sequence[Stage] = STAGES,
    base_python: str = sys.executable,
    offline: bool = False,
    demo_source: Path | None = None,
    demo_cache: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Install the wheel into a clean venv and run the stages against it.

    Args:
        wheelhouse: Directory holding the wheel.
        venv_dir: Directory for the clean venv, replaced if it exists.
        work_dir: Scratch directory for the stages.
        stages: Which stages to run.
        base_python: Interpreter the venv is made from — the matrix axis
            (spec §9).
        offline: Resolve the wheel's dependencies from the wheelhouse
            instead of an index.
        demo_source: A DOLFINx source tree to take the demos from.
        demo_cache: Where to download the pinned release instead.
        environ: Environment to derive the stages' from. Defaults to this
            process's.

    Returns:
        The wheel that was proven.

    Raises:
        FileNotFoundError: When the wheelhouse holds no wheel of ours, or
            the venv has no MPI launcher in it.
        ValueError: When it holds more than one.
        subprocess.CalledProcessError: When a stage fails. Its own message
            is on the standard error of this process, having been written
            there by the stage.
    """
    wheel = environment.find_wheel(wheelhouse)
    python = environment.create(
        venv_dir,
        wheel,
        base_python=base_python,
        wheelhouse=wheelhouse if offline else None,
    )
    site = environment.site_packages(python)
    launcher = environment.launcher(python)
    child = environment.child_environment(environ)

    for stage in stages:
        print(f"==> {stage.name}", flush=True)
        check_call(
            command(
                stage,
                python=python,
                site=site,
                launcher=launcher,
                work_dir=work_dir,
                demo_source=demo_source,
                demo_cache=demo_cache,
            ),
            env=child,
        )
    return wheel


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the wheel test suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        default=environment.DEFAULT_WHEELHOUSE,
        help="directory holding the wheel to prove",
    )
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        action="append",
        dest="stages",
        help=f"run one stage instead of all of them: "
        f"{', '.join(stage.name for stage in STAGES)}",
    )
    parser.add_argument(
        "--base-python",
        default=sys.executable,
        help="interpreter the clean venv is made from; the abi3 wheel has to "
        "install and work on 3.12, 3.13 and 3.14 (spec §9)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="resolve the wheel's dependencies from the wheelhouse, which "
        "then has to hold mpich, mpi4py, numpy, cffi, packaging and the "
        "upstream trio; nothing is fetched, not even a pip upgrade",
    )
    parser.add_argument("--demo-source", type=Path)
    parser.add_argument("--demo-cache", type=Path)
    args = parser.parse_args(argv)

    try:
        stages = [stage_by_name(name) for name in args.stages or []] or list(STAGES)
        wheel = run(
            wheelhouse=args.wheelhouse,
            venv_dir=args.venv,
            work_dir=args.work_dir,
            stages=stages,
            base_python=args.base_python,
            offline=args.offline,
            demo_source=args.demo_source,
            demo_cache=args.demo_cache or args.work_dir / "demo-source",
        )
    except (FileNotFoundError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    print(
        f"{wheel.name} proven on {Path(args.base_python).name}: "
        + "; ".join(stage.proves for stage in stages)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
