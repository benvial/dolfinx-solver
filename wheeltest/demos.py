"""Run upstream's own demos against the wheel.

The rest of the suite checks what this packaging decided: the import order,
the vendored linkage, the scalar type, two problems with known answers.
Upstream's demos check something the suite cannot write for itself — that
code written against DOLFINx, by the people who wrote DOLFINx, without any
knowledge of this wheel, runs in it. They reach corners a hand-written smoke
test has no reason to visit: mixed elements, XDMF and VTX output, boundary
condition machinery, the ffcx JIT compiling forms on the user's machine
against headers that come out of three different wheels.

Five of them are run (spec §11 asks for three to five), chosen so that each
covers something the others do not:

* three run under ``mpiexec -n 2``, which is where a demo meets the
  partitioner and the distributed assembly,
* two are written in the scalars this distribution is built for, which is
  why the subset follows :data:`wheelbuild.petsc.SCALAR_TYPE`: upstream's
  complex demos interpolate ``exp(1j...)`` without asking what
  ``PetscScalar`` is, and under a real build numpy drops the imaginary part
  with a warning and the demo exits zero having solved something else. A
  stage that passes for that reason proves less than nothing,
* one writes both ADIOS2 and XDMF output, which is the vendored I/O stack
  and its parallel HDF5 underneath it.

Demos that need something outside the wheel are not in the subset, and the
reason is recorded against each: this is a test of the wheel, and a demo
skipped for want of gmsh or matplotlib would be reported as a wheel that
cannot do something it can. The optional visualisation imports the chosen
demos do carry are all inside ``try``/``except ImportError`` upstream, so
they print a line about pyvista and carry on.

The sources come from the DOLFINx release this wheel mirrors, which is not
in the wheel: either from a source tree the caller already has (the
container build leaves one in its cache) or from the pinned tarball
:mod:`wheelbuild.dolfinx` names, fetched once into a cache directory and
checked against the digest that driver records for it (spec §10).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from wheelbuild import dolfinx as dolfinx_driver
from wheelbuild import petsc, sources
from wheeltest import environment

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


class Demo(NamedTuple):
    """One upstream demo, and why it is in the subset.

    Attributes:
        name: File name inside upstream's ``python/demo``.
        ranks: How many ranks to run it under.
        covers: What it proves that the others do not.
    """

    name: str
    ranks: int
    covers: str


#: The demos both variants run, as named constants so the two subsets below
#: can order them among their own.
POISSON = Demo(
    name="demo_poisson.py",
    ranks=2,
    covers=(
        "the canonical first program: a distributed mesh, a "
        "LinearProblem solved through the PETSc layer whose module body "
        "loads libpetsc by path (ticket 16), and XDMF output"
    ),
)
ELASTICITY = Demo(
    name="demo_elasticity.py",
    ranks=2,
    covers=(
        "a vector-valued problem assembled and solved in parallel with "
        "a near-nullspace attached, which is PETSc's own algebraic "
        "machinery rather than DOLFINx's"
    ),
)
INTERPOLATION_IO = Demo(
    name="demo_interpolation-io.py",
    ranks=2,
    covers=(
        "interpolation between spaces and VTX output through the "
        "vendored ADIOS2, in parallel — the I/O half of the stack, "
        "which no solve touches"
    ),
)

#: The two the complex variant adds.
HELMHOLTZ = Demo(
    name="demo_helmholtz.py",
    ranks=1,
    covers=(
        "a complex-scalar problem written by upstream, which takes its "
        "complex branch only when PETSc.ScalarType is complex — this "
        "distribution's reason to exist (spec §6)"
    ),
)
WAVEGUIDE = Demo(
    name="demo_half-loaded-waveguide.py",
    ranks=1,
    covers=(
        "a SLEPc eigenvalue problem over mixed Nedelec and Lagrange "
        "elements, complex-scalar: upstream code crossing the nanobind "
        "boundary into the fenics-basix wheel, which is the failure "
        "mode a version mismatch produces (ticket 05)"
    ),
)

#: The two the real variant adds in their place. Upstream has no real-scalar
#: eigenvalue demo, so slepc4py is left to :mod:`wheeltest.smoke`, which
#: solves the same spectrum under either variant; what these two buy instead
#: is form machinery and a mixed-element direct solve that no other demo in
#: the subset reaches.
BIHARMONIC = Demo(
    name="demo_biharmonic.py",
    ranks=1,
    covers=(
        "a fourth-order problem posed with a C0 interior-penalty method, "
        "whose interior-facet integrals are a corner of the form compiler "
        "no other demo in the subset visits"
    ),
)
STOKES = Demo(
    name="demo_stokes.py",
    ranks=2,
    covers=(
        "a mixed-element saddle-point problem solved in parallel, both "
        "monolithically through a vendored direct factorisation and "
        "block-wise through a fieldsplit preconditioner"
    ),
)

#: What each variant runs, ordered cheapest first, so a wheel that is broken
#: outright says so in seconds rather than after the last problem. The keys
#: are :data:`wheelbuild.petsc.SCALAR_TYPES`: two of upstream's demos are
#: written for complex scalars and cannot stand in for a real build (see this
#: module's docstring), so the subset is part of what the variant flip moves.
DEMO_SUBSETS = {
    "complex": (POISSON, HELMHOLTZ, ELASTICITY, INTERPOLATION_IO, WAVEGUIDE),
    "real": (POISSON, BIHARMONIC, ELASTICITY, INTERPOLATION_IO, STOKES),
}


def subset(variant: str = petsc.SCALAR_TYPE) -> tuple[Demo, ...]:
    """Return the demos one scalar variant is proved by.

    Args:
        variant: The scalar type this distribution is built for. It comes
            from the PETSc driver, so the flip that makes
            ``dolfinx-solver-real`` moves the subset with it.

    Returns:
        The demos, cheapest first.

    Raises:
        ValueError: When ``variant`` is not one of the two this build has a
            name for.
    """
    try:
        return DEMO_SUBSETS[variant]
    except KeyError:
        raise ValueError(
            f"{variant!r} is not a scalar variant this build knows: "
            f"{', '.join(petsc.SCALAR_TYPES)} (spec §6)."
        ) from None


#: The subset this build's variant runs.
DEMOS = subset()

#: What upstream calls the directory the demos live in, inside its tarball.
DEMO_RELATIVE = Path("python") / "demo"

#: Written inside an extracted tree, holding the digest it was extracted
#: from. The same marker ``scripts/build-wheel.sh`` writes, for the same
#: reason: the directory is named after the release, so the name alone cannot
#: notice that the release's bytes were replaced.
EXTRACTED_MARKER = ".extracted"

#: The metadata lookup the payload cannot answer. DOLFINx's own
#: ``__init__`` used to read its version this way and the assembly rewrites
#: it (ticket 14); a demo that asks the metadata layer directly would fail
#: for a packaging reason in the middle of a numerical test, so the subset
#: is scanned for it and the failure is reported as what it is.
METADATA_LOOKUP = re.compile(r"""version\(\s*["']fenics-dolfinx["']\s*\)""")


def demo_dir(source: Path) -> Path:
    """Return the demo directory inside a DOLFINx source tree.

    Args:
        source: An extracted source tree, or the demo directory itself.

    Returns:
        The directory holding the demo programs.

    Raises:
        FileNotFoundError: When neither is what was given.
    """
    if (source / DEMO_RELATIVE).is_dir():
        return source / DEMO_RELATIVE
    if (source / DEMOS[0].name).is_file():
        return source
    raise FileNotFoundError(
        f"{source} is neither a DOLFINx source tree (which would have "
        f"{DEMO_RELATIVE}) nor a directory of demo programs (which would "
        f"have {DEMOS[0].name})."
    )


def fetch(cache: Path, url: str = "", expected: str = "") -> Path:
    """Return the demo directory, downloading the pinned release if needed.

    The release is the one this wheel mirrors, read from the same driver the
    container build reads it from: running a newer release's demos against
    this wheel would be testing upstream's API drift, not the wheel.

    It is also the same tarball, at the same URL, as the build's DOLFINx
    stage, so it is verified against the same recorded digest rather than a
    second one kept in step by hand (ticket 10). These bytes become a test
    input rather than a vendored binary, which makes this the weaker of the
    two cases — but a demo stage that passed against a tarball nobody
    recognised would be proving something about the wrong sources.

    Args:
        cache: Directory to download and extract into, reused across runs.
        url: Where to fetch from. Defaults to the pinned source URL.
        expected: The SHA-256 to require. Defaults to the driver's.

    Returns:
        The demo directory.

    Raises:
        ValueError: When the URL is not an HTTPS one, or the archive is not
            the pinned one.
    """
    url = url or dolfinx_driver.source_url()
    expected = expected or dolfinx_driver.DOLFINX_SHA256
    if not url.startswith("https://"):
        raise ValueError(f"{url} is not an https URL, and the sources are.")
    cache.mkdir(parents=True, exist_ok=True)
    extracted = cache / dolfinx_driver.source_dir_name()
    marker = extracted / EXTRACTED_MARKER
    unpacked = (extracted / DEMO_RELATIVE).is_dir()
    verified = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    if not unpacked or verified != expected:
        # Nothing re-hashes an extracted tree, so the marker is what carries
        # the digest forward. Without it a digest moved without the version --
        # the re-rolled tarball this check exists for -- would keep running
        # the demos out of the bytes it replaced, because the directory name
        # is the release and the release did not change.
        shutil.rmtree(extracted, ignore_errors=True)
        # Verified on the run that unpacks it, not only on the run that
        # downloads it: the cache is kept between CI runs, so an archive
        # already sitting there is hashed too.
        archive = sources.fetch(
            url, cache / f"{dolfinx_driver.source_dir_name()}.tar.gz", expected
        )
        with tarfile.open(archive) as tar:
            tar.extractall(cache, filter="data")
        marker.write_text(f"{expected}\n", encoding="utf-8")
    return demo_dir(extracted)


def metadata_problem(sources: Iterable[tuple[str, str]]) -> str | None:
    """Report a demo that asks the metadata layer for the DOLFINx version.

    Args:
        sources: Each demo's name and its text.

    Returns:
        A message, or ``None`` when none of them does.
    """
    asking = [name for name, text in sources if METADATA_LOOKUP.search(text)]
    if not asking:
        return None
    return (
        f"{', '.join(asking)} reads the DOLFINx version from installed "
        "distribution metadata. This wheel registers no fenics-dolfinx "
        "distribution — a wheel may carry only one .dist-info and it is not "
        "ours to register — so that lookup raises PackageNotFoundError "
        "(ticket 14). The demo would fail for a packaging reason in the "
        "middle of a numerical test, which is worth saying plainly instead."
    )


def child_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the environment a demo runs in.

    The suite puts this repository on ``PYTHONPATH`` so its own stages can
    be imported. A demo is upstream code that knows nothing about this
    repository and has to run the way a user runs it, so it comes off again
    here.

    Args:
        environ: Environment to derive it from. Defaults to this process's.

    Returns:
        The environment.
    """
    child = dict(os.environ if environ is None else environ)
    child.pop("PYTHONPATH", None)
    return child


def command(demo: Demo, directory: Path, python: Path, launcher: Path) -> list[str]:
    """Return the command that runs one demo.

    Args:
        demo: The demo to run.
        directory: Where its sources are.
        python: Interpreter the wheel is installed for.
        launcher: The ``mpiexec`` beside it.

    Returns:
        The argument vector, under the launcher when the demo is a parallel
        one.
    """
    program = [str(python), str(directory / demo.name)]
    if demo.ranks > 1:
        return [str(launcher), "-n", str(demo.ranks), *program]
    return program


def outcome_problem(demo: Demo, returncode: int, output: str) -> str | None:
    """Report a demo that did not run.

    Args:
        demo: The demo that was run.
        returncode: What it exited with.
        output: What it wrote to its standard error.

    Returns:
        A message, or ``None`` when it ran.
    """
    if returncode == 0:
        return None
    return (
        f"{demo.name} failed under {demo.ranks} rank(s) with exit status "
        f"{returncode}. It is in the subset for {demo.covers}. Its output "
        f"ends:\n{_tail(output)}"
    )


def _tail(output: str, lines: int = 20) -> str:
    """Return the last lines of a program's output.

    Args:
        output: Everything it wrote.
        lines: How many lines to keep.

    Returns:
        The tail, which is where a traceback's message is.
    """
    return "\n".join(output.strip().splitlines()[-lines:])


def run_demo(
    demo: Demo, directory: Path, work_dir: Path, python: Path, launcher: Path
) -> str | None:
    """Run one demo in a directory of its own.

    Upstream's demos write their output into the working directory, so each
    gets one of its own: run in the source tree they would litter it, and a
    cached source tree that has been written into is a cache whose contents
    nobody can predict. The sources themselves are read by absolute path and
    never copied.

    Args:
        demo: The demo to run.
        directory: Where the demo sources are.
        work_dir: Directory to run it in.
        python: Interpreter the wheel is installed for.
        launcher: The ``mpiexec`` beside it.

    Returns:
        A message, or ``None`` when it ran.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command(demo, directory, python, launcher),
        cwd=work_dir,
        env=child_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    return outcome_problem(demo, completed.returncode, completed.stderr)


def check(
    directory: Path,
    work_dir: Path,
    python: Path,
    launcher: Path,
    demos: Sequence[Demo] = DEMOS,
) -> str | None:
    """Run the subset and report the first demo that did not.

    Args:
        directory: Where the demo sources are.
        work_dir: Directory to run them in, one subdirectory each.
        python: Interpreter the wheel is installed for.
        launcher: The ``mpiexec`` beside it.
        demos: The subset.

    Returns:
        A message, or ``None`` when every demo ran.

    Raises:
        FileNotFoundError: When a demo in the subset is not in the tree,
            which means upstream has renamed or removed it and the subset
            needs revisiting rather than silently shrinking.
    """
    missing = [demo.name for demo in demos if not (directory / demo.name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not in {directory}. The subset names "
            "upstream demos at the release this wheel mirrors; one that is "
            "not there means the subset has to be chosen again, not that "
            "there is less to test."
        )

    problem = metadata_problem(
        (demo.name, (directory / demo.name).read_text(encoding="utf-8"))
        for demo in demos
    )
    if problem is not None:
        return problem

    for demo in demos:
        print(f"+ demo {demo.name} on {demo.ranks} rank(s)", flush=True)
        problem = run_demo(
            demo, directory, work_dir / demo.name.removesuffix(".py"), python, launcher
        )
        if problem is not None:
            return problem
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the demo subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        help="a DOLFINx source tree to take the demos from; downloaded into "
        "--cache when not given",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(".build-cache") / "demo-source",
        help="where the pinned release is downloaded and extracted",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--scalar-type",
        default=petsc.SCALAR_TYPE,
        choices=list(petsc.SCALAR_TYPES),
        help="the variant the wheel under test was built for, which decides "
        "which demos are upstream's for it (spec §6)",
    )
    args = parser.parse_args(argv)

    chosen = subset(args.scalar_type)
    python = Path(sys.executable)
    launcher = environment.launcher(python)
    directory = demo_dir(args.source) if args.source else fetch(args.cache)

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    problem = check(directory, args.work_dir, python, launcher, chosen)
    if problem is not None:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    parallel = sum(1 for demo in chosen if demo.ranks > 1)
    print(
        f"{len(chosen)} upstream demos from {directory} ran against the "
        f"{args.scalar_type}-scalar payload in {args.site}, {parallel} of "
        f"them on {max(demo.ranks for demo in chosen)} ranks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
