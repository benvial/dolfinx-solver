"""Running the compilers and inspection tools the build steps drive.

Every build step is a long sequence of external commands whose output is the
only record of what happened, so commands are echoed before they run: a CI log
of a multi-hour container build has to be readable afterwards by someone who
was not watching it.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


def check_call(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Echo a command and run it, raising if it fails.

    Args:
        command: Argument vector to run.
        cwd: Directory to run it in. Defaults to the current one.
        env: Complete environment for the command. Defaults to inheriting.

    Raises:
        subprocess.CalledProcessError: When the command exits non-zero.
    """
    print("+ " + " ".join(command), flush=True)
    subprocess.check_call(
        list(command),
        cwd=None if cwd is None else str(cwd),
        env=None if env is None else dict(env),
    )


def capture(command: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run a command and return its standard output.

    Used for the inspection tools — ``nm``, ``mpichversion`` — whose output is
    the thing being checked rather than progress to watch.

    Args:
        command: Argument vector to run.
        cwd: Directory to run it in. Defaults to the current one.

    Returns:
        Everything the command wrote to standard output, decoded as text.

    Raises:
        subprocess.CalledProcessError: When the command exits non-zero.
    """
    print("+ " + " ".join(command), flush=True)
    return subprocess.check_output(
        list(command),
        cwd=None if cwd is None else str(cwd),
        text=True,
    )
