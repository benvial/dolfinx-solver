"""Harvest the licence text of everything the wheel carries.

A wheel is a redistribution. Every library inside it was written by someone
else and is shipped under someone else's terms, and most of those terms —
BSD-3, MIT, Apache-2.0, the CeCILL-C of MUMPS and SCOTCH, the LGPL of DOLFINx
itself — require the licence text and the copyright notice to travel with the
binary. Spec §7 therefore makes the notice file build-enforced rather than
hand-maintained: the build fails when a component this wheel ships has no
harvested licence text, which is the only way a notice file stays true across
a version bump nobody remembered to re-read.

Enforcement runs in both directions, and the second one is the interesting
half:

- **Every component needs a text.** :data:`COMPONENTS` says where each
  component's licence lives, as a glob against one of three roots (an unpacked
  source tree, PETSc's ``--download-*`` tree, the build image's own
  ``/usr/share/licenses``). A glob that matches nothing fails the build, so a
  source tree that renames ``LICENSE`` to ``LICENSE.md`` is caught here rather
  than discovered by a user.
- **Every shipped library needs a component.** Each entry also declares the
  sonames it accounts for, and :func:`unclaimed_libraries` reports any library
  in the assembled wheel that no entry claims. That is the check that catches
  the library nobody chose: a new ``--download-*`` package, a transitive
  dependency auditwheel grafted off the build image, a bundled third-party
  library a release started installing. Without it the notice file would
  quietly describe a wheel that no longer exists.

Some components ship no shared library of their own and are on the list all
the same, because their code is inside one that does: MUMPS installs static
archives that are linked into ``libpetsc``, Boost is header-only, and ADIOS2
compiles bundled copies of pugixml, nlohmann/json, yaml-cpp and KWSys into its
own libraries. A harvester that walked the shipped ``.so`` files alone would
miss every one of them, which is why the component list is written down rather
than discovered.

Two of those bundled components keep their notice inside a source file instead
of a licence file of their own; :class:`Text` carries a marker for that case
and :func:`comment_block` lifts the comment the marker sits in.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from wheelbuild import adios2, kahip, mpich, petsc, slepc, version

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: The three places a licence text can be read from.
#:
#: ``source`` is an unpacked source tarball under the build root, which is
#: where the tarballs this build downloads itself are extracted. ``downloaded``
#: is PETSc's ``externalpackages`` directory, where the seven packages PETSc's
#: configure fetches end up — their directory names carry versions PETSc
#: chooses, not ones this repository pins, which is why those entries are
#: globs. ``image`` is the build image's ``/usr/share/licenses``, for the
#: libraries that arrive as RPMs rather than as source: pugixml, spdlog, fmt,
#: Boost's headers and the GCC runtime libraries auditwheel grafts.
SOURCE = "source"
DOWNLOADED = "downloaded"
IMAGE = "image"


class Roots(NamedTuple):
    """Where the three kinds of licence text are looked for.

    Attributes:
        source: Build root holding the unpacked source tarballs.
        downloaded: PETSc's ``externalpackages`` directory.
        image: The build image's licence directory.
    """

    source: Path
    downloaded: Path
    image: Path

    def directory(self, root: str) -> Path:
        """Return the directory one root name stands for.

        Args:
            root: One of :data:`SOURCE`, :data:`DOWNLOADED`, :data:`IMAGE`.

        Returns:
            The directory that root resolves to.

        Raises:
            KeyError: When the name is not one of the three.
        """
        return {
            SOURCE: self.source,
            DOWNLOADED: self.downloaded,
            IMAGE: self.image,
        }[root]


def roots(
    build_root: Path,
    *,
    petsc_source: Path | None = None,
    scalar_type: str = petsc.SCALAR_TYPE,
    image_licenses: Path = Path("/usr/share/licenses"),
) -> Roots:
    """Return the roots a container build harvests from.

    Args:
        build_root: Directory the source tarballs are unpacked into.
        petsc_source: PETSc's unpacked source tree. Defaults to the one the
            PETSc driver's pinned release unpacks to, under ``build_root``.
        scalar_type: PETSc scalar type, which names the arch directory its
            downloaded packages live under.
        image_licenses: The build image's licence directory.

    Returns:
        The roots :func:`harvest` reads through.
    """
    if petsc_source is None:
        petsc_source = build_root / f"petsc-{petsc.PETSC_VERSION}"
    return Roots(
        source=build_root,
        downloaded=petsc_source / petsc.build_arch(scalar_type) / "externalpackages",
        image=image_licenses,
    )


class Text(NamedTuple):
    """One licence text to harvest.

    Attributes:
        root: Which of the three roots the pattern is relative to.
        pattern: Glob for the file, relative to that root. It has to match
            exactly one file: no match is a component whose licence this build
            cannot find, and several is an ambiguity that should be spelled out
            rather than guessed at.
        marker: When set, the text is not the whole file but the comment block
            containing this line of it — see :func:`comment_block`. Used for
            the two bundled components that keep their notice inside a source
            header instead of a licence file.
    """

    root: str
    pattern: str
    marker: str | None = None


class Component(NamedTuple):
    """One third-party component this wheel redistributes.

    Attributes:
        name: How the component is named in the notice file.
        license_id: Its licence, as an SPDX expression.
        texts: Where its licence text is read from. All of them are harvested:
            the GCC runtime libraries need both the licence and the runtime
            exception to make sense, and HDF5 ships two.
        libraries: Soname patterns this component accounts for, matched with
            :func:`fnmatch.fnmatch` against the libraries the assembled wheel
            carries. Empty for a component whose code ships inside another
            library — a static archive, a header-only library, a bundled
            source tree — which is most of the interesting cases.
        corresponding_source: Where the source of this binary can be had.
            Required by the copyleft licences in the stack (the LGPL of
            DOLFINx, the CeCILL-C of MUMPS and SCOTCH, the GPL-with-exception
            of the GCC runtime), and recorded for the permissive ones too
            because a notice file is more useful with it than without.
        note: One sentence for the reader when what the component is, or why
            it is here at all, is not obvious from its name.
    """

    name: str
    license_id: str
    texts: tuple[Text, ...]
    libraries: tuple[str, ...] = ()
    corresponding_source: str | None = None
    note: str | None = None


#: Every component the wheel ships, and where its licence text comes from.
#:
#: The order is the order the notice file reads in: the product, then the
#: solver stack, then the libraries that arrive with the build image. Sonames
#: are patterns because most of this stack versions its libraries and because
#: auditwheel renames what it grafts — ``libgfortran.so.5`` is copied in as
#: ``libgfortran-a1b2c3d4.so.5``, so the pattern has to allow the hash.
COMPONENTS = (
    Component(
        name="DOLFINx",
        license_id="LGPL-3.0-or-later",
        texts=(
            Text(SOURCE, f"dolfinx-{version.DOLFINX_VERSION}/COPYING.LESSER"),
            Text(SOURCE, f"dolfinx-{version.DOLFINX_VERSION}/COPYING"),
        ),
        libraries=("libdolfinx.so.*",),
        corresponding_source=(
            f"https://github.com/FEniCS/dolfinx/releases/tag/v{version.DOLFINX_VERSION}"
        ),
        note=(
            "The library this wheel exists to ship, with its Python bindings "
            "in the dolfinx package beside this file."
        ),
    ),
    Component(
        name="PETSc",
        license_id="BSD-2-Clause",
        texts=(Text(SOURCE, f"petsc-{petsc.PETSC_VERSION}/LICENSE"),),
        libraries=("libpetsc.so.*",),
        corresponding_source=(
            f"https://gitlab.com/petsc/petsc/-/tree/v{petsc.PETSC_VERSION}"
        ),
        note="Built for complex scalars; see the petsc4py entry for the bindings.",
    ),
    Component(
        name="petsc4py",
        license_id="BSD-2-Clause",
        texts=(
            Text(
                SOURCE,
                f"petsc-{petsc.PETSC_VERSION}/src/binding/petsc4py/LICENSE.rst",
            ),
        ),
        note=(
            "Built here against the vendored PETSc and shipped under its "
            "upstream import name; the PyPI distribution of the same name is "
            "a different build (spec §6)."
        ),
    ),
    Component(
        name="SLEPc",
        license_id="BSD-2-Clause",
        texts=(Text(SOURCE, f"slepc-{slepc.SLEPC_VERSION}/LICENSE.md"),),
        libraries=("libslepc.so.*",),
        corresponding_source=(
            f"https://gitlab.com/slepc/slepc/-/tree/v{slepc.SLEPC_VERSION}"
        ),
    ),
    Component(
        name="slepc4py",
        license_id="BSD-2-Clause",
        texts=(
            Text(
                SOURCE,
                f"slepc-{slepc.SLEPC_VERSION}/src/binding/slepc4py/LICENSE.rst",
            ),
        ),
    ),
    Component(
        name="OpenBLAS",
        license_id="BSD-3-Clause",
        texts=(Text(DOWNLOADED, "git.openblas/LICENSE"),),
        libraries=("libopenblas*.so.*", "libopenblas*.so"),
        corresponding_source="https://github.com/OpenMathLib/OpenBLAS",
        note=(
            "Built with DYNAMIC_ARCH, so it dispatches on the running "
            "machine rather than the one that compiled the wheel."
        ),
    ),
    Component(
        name="ScaLAPACK",
        license_id="BSD-3-Clause",
        texts=(Text(DOWNLOADED, "git.scalapack/LICENSE"),),
        libraries=("libscalapack.so.*",),
        corresponding_source="https://github.com/Reference-ScaLAPACK/scalapack",
    ),
    Component(
        name="METIS",
        license_id="Apache-2.0",
        texts=(Text(DOWNLOADED, "git.metis/LICENSE.txt"),),
        libraries=("libmetis.so*",),
        corresponding_source="https://bitbucket.org/petsc/pkg-metis",
        note=(
            "The Apache-2.0 KarypisLab release, used for MUMPS's serial "
            "ordering. ParMETIS is not in this wheel (spec §7)."
        ),
    ),
    Component(
        name="SCOTCH and PT-SCOTCH",
        license_id="CECILL-C",
        texts=(
            Text(DOWNLOADED, "git.ptscotch/LICENSE_en.txt"),
            Text(DOWNLOADED, "git.ptscotch/LICENCE_fr.txt"),
        ),
        libraries=(
            "libscotch.so.*",
            "libscotcherr*.so.*",
            "libptscotch.so.*",
            "libptscotcherr*.so.*",
            "libptscotchparmetisv3.so.*",
            "libesmumps.so.*",
            "libptesmumps.so.*",
        ),
        corresponding_source="https://gitlab.inria.fr/scotch/scotch",
        note=(
            "The wheel's parallel partitioner, and MUMPS's parallel ordering. "
            "libptscotchparmetisv3 is SCOTCH's own emulation of the ParMETIS "
            "interface under this licence, not ParMETIS."
        ),
    ),
    Component(
        name="MUMPS",
        license_id="CECILL-C",
        texts=(Text(DOWNLOADED, "MUMPS_*/LICENSE"),),
        corresponding_source="https://mumps-solver.org/index.php?page=dwnld",
        note=(
            "Installed as static archives and linked into libpetsc, so it "
            "ships inside that library rather than as one of its own."
        ),
    ),
    Component(
        name="SuperLU_DIST",
        license_id="BSD-3-Clause",
        texts=(Text(DOWNLOADED, "git.superlu_dist/License.txt"),),
        libraries=("libsuperlu_dist.so.*",),
        corresponding_source="https://github.com/xiaoyeli/superlu_dist",
    ),
    Component(
        name="HDF5",
        license_id="BSD-3-Clause",
        texts=(
            Text(DOWNLOADED, "hdf5-*/COPYING"),
            Text(DOWNLOADED, "hdf5-*/COPYING_LBNL_HDF5"),
        ),
        libraries=("libhdf5.so.*", "libhdf5_hl.so.*"),
        corresponding_source="https://github.com/HDFGroup/hdf5",
        note="Built parallel, against the MPI this wheel links.",
    ),
    Component(
        name="ADIOS2",
        license_id="Apache-2.0",
        texts=(
            Text(SOURCE, f"ADIOS2-{adios2.ADIOS2_VERSION}/LICENSE"),
            Text(SOURCE, f"ADIOS2-{adios2.ADIOS2_VERSION}/Copyright.txt"),
        ),
        libraries=(
            "libadios2_core.so.*",
            "libadios2_core_mpi.so.*",
            "libadios2_cxx.so.*",
            "libadios2_cxx_mpi.so.*",
        ),
        corresponding_source=(
            f"https://github.com/ornladios/ADIOS2/releases/tag/v{adios2.ADIOS2_VERSION}"
        ),
        note="Backs DOLFINx's VTX and checkpoint I/O.",
    ),
    Component(
        name="ATL (bundled in ADIOS2)",
        license_id="BSD-3-Clause",
        texts=(
            Text(SOURCE, f"ADIOS2-{adios2.ADIOS2_VERSION}/thirdparty/atl/atl/LICENSE"),
        ),
        libraries=("libadios2_atl.so.*",),
        note="ADIOS2's attribute library, needed by its BP engines.",
    ),
    Component(
        name="DILL (bundled in ADIOS2)",
        license_id="BSD-3-Clause",
        texts=(
            Text(
                SOURCE, f"ADIOS2-{adios2.ADIOS2_VERSION}/thirdparty/dill/dill/LICENSE"
            ),
        ),
        libraries=("libadios2_dill.so.*",),
        note="FFS's code generator.",
    ),
    Component(
        name="FFS (bundled in ADIOS2)",
        license_id="BSD-3-Clause",
        texts=(
            Text(SOURCE, f"ADIOS2-{adios2.ADIOS2_VERSION}/thirdparty/ffs/ffs/LICENSE"),
        ),
        libraries=("libadios2_ffs.so.*",),
        note="The marshalling library ADIOS2's core links.",
    ),
    Component(
        name="PerfStubs (bundled in ADIOS2)",
        license_id="BSD-3-Clause",
        texts=(
            Text(
                SOURCE,
                f"ADIOS2-{adios2.ADIOS2_VERSION}/thirdparty/perfstubs/perfstubs/LICENSE",
            ),
        ),
        libraries=("libadios2_perfstubs.so.*",),
        note="ADIOS2's instrumentation shim, linked by its core.",
    ),
    Component(
        name="KWSys (bundled in ADIOS2)",
        license_id="BSD-3-Clause",
        texts=(
            Text(
                SOURCE,
                f"ADIOS2-{adios2.ADIOS2_VERSION}/thirdparty/KWSys/adios2sys/Copyright.txt",
            ),
        ),
        note="Compiled into ADIOS2's own libraries; ships no library of its own.",
    ),
    Component(
        name="yaml-cpp (bundled in ADIOS2)",
        license_id="MIT",
        texts=(
            Text(
                SOURCE,
                f"ADIOS2-{adios2.ADIOS2_VERSION}/thirdparty/yaml-cpp/yaml-cpp/LICENSE",
            ),
        ),
        note="Compiled into ADIOS2's own libraries.",
    ),
    Component(
        name="pugixml (bundled in ADIOS2)",
        license_id="MIT",
        texts=(
            Text(
                SOURCE,
                f"ADIOS2-{adios2.ADIOS2_VERSION}/thirdparty/pugixml/pugixml/src/pugixml.hpp",
                marker="Permission is hereby granted, free of charge",
            ),
        ),
        note=(
            "Compiled into ADIOS2's own libraries, and a separate copy from "
            "the libpugixml DOLFINx links. Its notice lives at the end of "
            "pugixml.hpp rather than in a licence file."
        ),
    ),
    Component(
        name="nlohmann/json (bundled in ADIOS2)",
        license_id="MIT",
        texts=(
            Text(
                SOURCE,
                f"ADIOS2-{adios2.ADIOS2_VERSION}/thirdparty/nlohmann_json/"
                "nlohmann_json_wrapper/single_include/nlohmann/json.hpp",
                marker="Permission is hereby  granted, free of charge",
            ),
        ),
        note=(
            "Header-only and compiled into ADIOS2's own libraries. Its notice "
            "is the comment at the top of json.hpp."
        ),
    ),
    Component(
        name="KaHIP",
        license_id="MIT",
        texts=(Text(SOURCE, f"KaHIP-{kahip.KAHIP_VERSION}/LICENSE"),),
        libraries=("libkahip.so*", "libparhip_interface.so*"),
        corresponding_source=(
            f"https://github.com/KaHIP/KaHIP/releases/tag/v{kahip.KAHIP_VERSION}"
        ),
        note="The optional second partitioner beside PT-SCOTCH.",
    ),
    Component(
        name="MPICH",
        license_id="LicenseRef-MPICH",
        texts=(Text(SOURCE, f"mpich-{mpich.MPICH_VERSION}/COPYRIGHT"),),
        libraries=("libmpifort.so.*", "libmpicxx.so.*"),
        corresponding_source=(
            f"https://www.mpich.org/static/downloads/{mpich.MPICH_VERSION}/"
            f"mpich-{mpich.MPICH_VERSION}.tar.gz"
        ),
        note=(
            "Only the Fortran and C++ binding halves are here. libmpi itself "
            "comes from the PyPI mpich wheel and is never vendored (spec §5)."
        ),
    ),
    Component(
        name="Boost",
        license_id="BSL-1.0",
        texts=(Text(IMAGE, "boost/LICENSE_1_0.txt"),),
        note=(
            "Header-only use by DOLFINx's graph code, so its code ships "
            "inside libdolfinx and no Boost library is in this wheel."
        ),
    ),
    Component(
        name="pugixml",
        license_id="MIT",
        texts=(Text(IMAGE, "pugixml/LICENSE.md"),),
        libraries=("libpugixml*.so.*",),
        corresponding_source="https://github.com/zeux/pugixml",
        note="DOLFINx's XML reader; from the build image, grafted by auditwheel.",
    ),
    Component(
        name="spdlog",
        license_id="MIT",
        texts=(Text(IMAGE, "spdlog/LICENSE"),),
        libraries=("libspdlog*.so.*",),
        corresponding_source="https://github.com/gabime/spdlog",
        note="DOLFINx's logger; from the build image, grafted by auditwheel.",
    ),
    Component(
        name="fmt",
        license_id="MIT",
        texts=(Text(IMAGE, "fmt/LICENSE.rst"),),
        libraries=("libfmt*.so.*",),
        corresponding_source="https://github.com/fmtlib/fmt",
        note="spdlog's formatting library, and here for the same reason it is.",
    ),
    Component(
        name="libgfortran (GCC runtime)",
        license_id="GPL-3.0-or-later WITH GCC-exception-3.1",
        texts=(
            Text(IMAGE, "gcc/COPYING.RUNTIME"),
            Text(IMAGE, "gcc/COPYING3"),
        ),
        libraries=("libgfortran*.so.*",),
        corresponding_source="https://gcc.gnu.org/",
        note=(
            "The Fortran runtime the Fortran halves of this stack — MUMPS, "
            "ScaLAPACK, PETSc's Fortran sources — need. The runtime library "
            "exception is what allows it to ship alongside non-GPL code."
        ),
    ),
    Component(
        name="libquadmath (GCC runtime)",
        license_id="LGPL-2.1-or-later",
        texts=(
            Text(IMAGE, "libquadmath/COPYING.LIB.libquadmath"),
            Text(IMAGE, "gcc/COPYING.RUNTIME"),
        ),
        libraries=("libquadmath*.so.*",),
        corresponding_source="https://gcc.gnu.org/",
        note="Quad-precision support for the same Fortran code.",
    ),
)


def comment_block(text: str, marker: str) -> str | None:
    """Return the C comment block containing a line, stripped of decoration.

    Two of the bundled components keep their licence notice inside a source
    header rather than in a file of their own. What is wanted is the notice,
    not the header, so the block the marker sits in is cut out of the file and
    the ``/*``, ``*/`` and leading ``*`` are removed.

    Args:
        text: Contents of the source file.
        marker: A distinctive substring of one line of the notice.

    Returns:
        The comment's text, or ``None`` when no comment contains the marker.
    """
    lines = text.splitlines()
    found = next((index for index, line in enumerate(lines) if marker in line), None)
    if found is None:
        return None

    start = next(
        (
            index
            for index in range(found, -1, -1)
            if lines[index].lstrip().startswith("/*")
        ),
        None,
    )
    end = next(
        (
            index
            for index in range(found, len(lines))
            if lines[index].rstrip().endswith("*/")
        ),
        None,
    )
    if start is None or end is None:
        return None

    body = []
    for line in lines[start : end + 1]:
        stripped = line.strip()
        for opening in ("/**", "/*"):
            if stripped.startswith(opening):
                stripped = stripped[len(opening) :]
                break
        stripped = stripped.removesuffix("*/")
        stripped = stripped.lstrip("*")
        body.append(stripped.strip())
    return "\n".join(body).strip()


def resolve(text: Text, where: Roots) -> Path:
    """Return the single file one licence text pattern names.

    Args:
        text: The text to locate.
        where: The roots to look in.

    Returns:
        The file's path.

    Raises:
        FileNotFoundError: When the pattern matches nothing — the build
            failure spec §7 asks for.
        ValueError: When it matches more than one file, which is an ambiguity
            to spell out rather than resolve by sorting.
    """
    root = where.directory(text.root)
    matches = sorted(root.glob(text.pattern))
    if not matches:
        raise FileNotFoundError(
            f"no licence text at {root / text.pattern}. Every component this "
            "wheel vendors ships its licence text with it (spec §7), so a "
            "pattern that matches nothing is a notice file that would be "
            "wrong rather than a step to skip: either the component moved its "
            "licence file, or this build no longer produces what the "
            "component list describes."
        )
    if len(matches) > 1:
        found = ", ".join(str(match.relative_to(root)) for match in matches)
        raise ValueError(
            f"{root / text.pattern} matches {len(matches)} files ({found}). "
            "A licence text pattern has to name one file; narrow it rather "
            "than letting the harvest pick."
        )
    return matches[0]


def read(text: Text, where: Roots) -> tuple[Path, str]:
    """Read one licence text.

    Args:
        text: The text to harvest.
        where: The roots to look in.

    Returns:
        The file it came from and the text itself.

    Raises:
        FileNotFoundError: When the pattern matches nothing.
        ValueError: When it matches several files, or when a marker finds no
            comment in the file it names.
    """
    path = resolve(text, where)
    contents = path.read_text(encoding="utf-8", errors="replace")
    if text.marker is None:
        return path, contents.strip()

    block = comment_block(contents, text.marker)
    if block is None:
        raise ValueError(
            f"{path} has no comment containing {text.marker!r}, so the licence "
            "notice this component keeps inside its source cannot be lifted "
            "out of it. The file has been reorganised upstream and the marker "
            "has to follow it."
        )
    return path, block


def claims(component: Component, soname: str) -> bool:
    """Report whether a component accounts for one shipped library.

    Args:
        component: The component to ask.
        soname: File name of a library in the assembled wheel.

    Returns:
        ``True`` when one of the component's patterns matches.
    """
    return any(fnmatch(soname, pattern) for pattern in component.libraries)


def unclaimed_libraries(
    sonames: Iterable[str], components: Sequence[Component] = COMPONENTS
) -> list[str]:
    """Return the shipped libraries no component accounts for.

    This is the direction of the check that catches what nobody chose: a new
    ``--download-*`` package, a bundled library a release started installing,
    a transitive dependency auditwheel grafted off the build image. Any of them
    is a library in a published wheel with no licence text beside it.

    Args:
        sonames: File names of the libraries the wheel carries.
        components: The component list to check against.

    Returns:
        The unaccounted-for names, sorted.
    """
    return sorted(
        {
            soname
            for soname in sonames
            if not any(claims(component, soname) for component in components)
        }
    )


#: How the notice file introduces itself. The wheel is LGPL-3.0-or-later and
#: the vendored stack is not, so the reader has to be told which terms apply to
#: what before the texts start.
PREAMBLE = """\
This file is a build artefact: it is assembled from the licence texts found in
the source trees this wheel was built from, and the build fails when a
component it ships has no text to harvest.

The dolfinx-solver-complex distribution itself is LGPL-3.0-or-later. It
vendors the libraries listed below, each under its own terms. Where a licence
requires the corresponding source to be available, the entry says where to get
the exact release this wheel was built from.

libmpi is deliberately *not* in this list: the wheel links the MPI runtime that
the PyPI mpich wheel installs, and vendors only MPICH's Fortran and C++ binding
libraries. fenics-basix, fenics-ffcx and fenics-ufl are runtime dependencies
installed as their own distributions, with their own notices, and are not
vendored here either.
"""

#: Width of the rules between entries, in characters.
_RULE = 78


def render(
    harvested: Sequence[tuple[Component, tuple[tuple[Path, str], ...]]],
    *,
    distribution_version: str = version.__version__,
    relative_to: Path | None = None,
) -> str:
    """Render the notice file from harvested texts.

    Args:
        harvested: Each component with the files and texts read for it.
        distribution_version: Version of the wheel being described.
        relative_to: Directory the recorded source paths are reported relative
            to, so the notice names ``petsc-3.25.5/LICENSE`` rather than a
            path inside a container. Absolute paths are reported when a file
            lies outside it — the build image's licences do.

    Returns:
        The complete notice file.
    """
    lines = [
        f"THIRD-PARTY NOTICES for dolfinx-solver-complex {distribution_version}",
        "",
        PREAMBLE,
    ]
    for component, texts in harvested:
        lines.extend(["=" * _RULE, f"{component.name} — {component.license_id}"])
        if component.note:
            lines.append(textwrap.fill(component.note, width=_RULE))
        if component.corresponding_source:
            lines.append(f"Corresponding source: {component.corresponding_source}")
        if component.libraries:
            lines.append(f"Accounts for: {', '.join(component.libraries)}")
        for path, text in texts:
            reported = path
            if relative_to is not None:
                try:
                    reported = path.relative_to(relative_to)
                except ValueError:
                    reported = path
            lines.extend(["-" * _RULE, f"Licence text from {reported}", "", text, ""])
    return "\n".join(lines) + "\n"


def harvest(
    where: Roots,
    *,
    components: Sequence[Component] = COMPONENTS,
    distribution_version: str = version.__version__,
) -> str:
    """Read every component's licence text and render the notice file.

    Args:
        where: The roots to harvest from.
        components: The component list to harvest.
        distribution_version: Version of the wheel being described.

    Returns:
        The complete notice file.

    Raises:
        FileNotFoundError: When a component's licence text is not where the
            list says it is.
        ValueError: When a pattern is ambiguous or a marker finds no comment.
    """
    harvested = [
        (component, tuple(read(text, where) for text in component.texts))
        for component in components
    ]
    return render(
        harvested,
        distribution_version=distribution_version,
        relative_to=where.source,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for the notice harvester.

    Writing the notice file is the assembly step's job; this entry point is
    for looking at what the harvest produces without building a wheel.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument(
        "--image-licenses",
        type=Path,
        default=Path("/usr/share/licenses"),
        help="the build image's licence directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="file to write the notices to; defaults to standard output",
    )
    args = parser.parse_args(argv)

    try:
        notices = harvest(roots(args.build_root, image_licenses=args.image_licenses))
    except (FileNotFoundError, ValueError) as problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    if args.output is None:
        print(notices)
    else:
        args.output.write_text(notices, encoding="utf-8")
        print(
            f"harvested {len(COMPONENTS)} components into {args.output} "
            f"({len(notices.splitlines())} lines)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
