"""The notice harvester: a missing licence text is a failed build."""

from pathlib import Path

import pytest

from wheelbuild import notices

PUGIXML_TAIL = """\
#endif

/**
 * Copyright (c) 2006-2025 Arseny Kapoulkine
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software.
 */
"""

JSON_HEAD = """\
/*
    JSON for Modern C++ version 3.10.5

Copyright (c) 2013-2022 Niels Lohmann.

Permission is hereby  granted, free of charge, to any  person obtaining a copy
of this software.
*/

#ifndef INCLUDE_NLOHMANN_JSON_HPP_
"""


def _roots(tmp_path: Path) -> notices.Roots:
    """Lay out the three roots a harvest reads through."""
    where = notices.Roots(
        source=tmp_path / "build",
        downloaded=tmp_path / "build" / "petsc" / "externalpackages",
        image=tmp_path / "licenses",
    )
    for directory in where:
        directory.mkdir(parents=True, exist_ok=True)
    return where


def _write(path: Path, text: str) -> Path:
    """Write a file, making its directory first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_the_roots_are_named_not_positional():
    """A pattern says which root it belongs to; the harvest resolves it."""
    where = notices.Roots(
        source=Path("/build"),
        downloaded=Path("/build/petsc/externalpackages"),
        image=Path("/usr/share/licenses"),
    )

    assert where.directory(notices.SOURCE) == Path("/build")
    assert where.directory(notices.IMAGE) == Path("/usr/share/licenses")
    with pytest.raises(KeyError):
        where.directory("somewhere-else")


def test_the_downloaded_root_is_petscs_arch_tree():
    """Every ``--download-*`` licence lives under PETSc's arch directory."""
    where = notices.roots(Path("/build"))

    assert where.source == Path("/build")
    assert where.downloaded.name == "externalpackages"
    assert "arch-complex" in where.downloaded.parts


def test_a_missing_licence_text_fails_the_harvest(tmp_path):
    """The check spec §7 asks for: no text, no wheel."""
    where = _roots(tmp_path)
    component = notices.Component(
        name="Somelib",
        license_id="MIT",
        texts=(notices.Text(notices.SOURCE, "somelib-1.0/LICENSE"),),
    )

    with pytest.raises(FileNotFoundError, match=r"somelib-1\.0/LICENSE"):
        notices.harvest(where, components=[component])


def test_an_ambiguous_pattern_fails_rather_than_picking(tmp_path):
    """Two matches mean the list is wrong, not that either will do."""
    where = _roots(tmp_path)
    _write(where.downloaded / "MUMPS_5.8.2" / "LICENSE", "cecill")
    _write(where.downloaded / "MUMPS_5.9.0" / "LICENSE", "cecill")
    component = notices.Component(
        name="MUMPS",
        license_id="CECILL-C",
        texts=(notices.Text(notices.DOWNLOADED, "MUMPS_*/LICENSE"),),
    )

    with pytest.raises(ValueError, match="matches 2 files"):
        notices.harvest(where, components=[component])


def test_a_glob_finds_the_version_petsc_chose(tmp_path):
    """PETSc names its download directories; this repository does not."""
    where = _roots(tmp_path)
    _write(where.downloaded / "hdf5-hdf5_1.14.6" / "COPYING", "the hdf5 licence")

    path, text = notices.read(notices.Text(notices.DOWNLOADED, "hdf5-*/COPYING"), where)

    assert path.parent.name == "hdf5-hdf5_1.14.6"
    assert text == "the hdf5 licence"


def test_a_marker_lifts_a_notice_out_of_a_source_header(tmp_path):
    """pugixml's notice is the comment at the end of pugixml.hpp."""
    where = _roots(tmp_path)
    _write(where.source / "pugixml.hpp", PUGIXML_TAIL)

    _, text = notices.read(
        notices.Text(
            notices.SOURCE, "pugixml.hpp", marker="Permission is hereby granted"
        ),
        where,
    )

    assert text.startswith("Copyright (c) 2006-2025 Arseny Kapoulkine")
    assert text.endswith("copy of this software.")
    assert "*" not in text
    assert "#endif" not in text


def test_a_marker_also_lifts_a_notice_written_without_comment_decoration(tmp_path):
    """nlohmann/json's block has no leading ``*`` on its lines."""
    where = _roots(tmp_path)
    _write(where.source / "json.hpp", JSON_HEAD)

    _, text = notices.read(
        notices.Text(
            notices.SOURCE, "json.hpp", marker="Permission is hereby  granted"
        ),
        where,
    )

    assert "Copyright (c) 2013-2022 Niels Lohmann." in text
    assert "INCLUDE_NLOHMANN_JSON_HPP_" not in text


def test_a_marker_that_no_longer_matches_fails_the_harvest(tmp_path):
    """An upstream reorganisation has to move the marker, not pass silently."""
    where = _roots(tmp_path)
    _write(where.source / "pugixml.hpp", "// no licence here\n")

    with pytest.raises(ValueError, match="no comment containing"):
        notices.read(
            notices.Text(notices.SOURCE, "pugixml.hpp", marker="Permission"), where
        )


def test_the_harvest_records_where_each_text_came_from(tmp_path):
    """The notice file names its own provenance, relative to the build root."""
    where = _roots(tmp_path)
    _write(where.source / "somelib-1.0" / "LICENSE", "do as you please")
    component = notices.Component(
        name="Somelib",
        license_id="MIT",
        texts=(notices.Text(notices.SOURCE, "somelib-1.0/LICENSE"),),
        libraries=("libsomelib.so.*",),
        corresponding_source="https://example.invalid/somelib",
        note="Here because something links it.",
    )

    rendered = notices.harvest(where, components=[component])

    assert "Somelib — MIT" in rendered
    assert "Licence text from somelib-1.0/LICENSE" in rendered
    assert "do as you please" in rendered
    assert "Corresponding source: https://example.invalid/somelib" in rendered
    assert "Here because something links it." in rendered


def test_the_preamble_says_libmpi_is_not_vendored():
    """The one library the wheel links and never ships (spec §5)."""
    assert "libmpi" in notices.PREAMBLE
    assert "fenics-basix" in notices.PREAMBLE


def test_every_component_declares_a_licence_text():
    """An entry with no text would be a component the harvest cannot check."""
    for component in notices.COMPONENTS:
        assert component.texts, component.name


def test_every_copyleft_component_points_at_its_corresponding_source():
    """Spec §7: LGPL, GPL and CeCILL-C need the pointer, not just the text."""
    copyleft = ("LGPL", "GPL", "CECILL")
    for component in notices.COMPONENTS:
        if component.license_id.upper().startswith(copyleft):
            assert component.corresponding_source, component.name


def test_the_component_names_are_distinct():
    """Two entries with one name read as a duplicate rather than two things."""
    names = [component.name for component in notices.COMPONENTS]
    assert len(names) == len(set(names))


def test_an_unclaimed_library_is_reported():
    """A library nobody chose is the failure this direction of the check exists for."""
    assert notices.unclaimed_libraries(["libpetsc.so.3.25"]) == []
    assert notices.unclaimed_libraries(["libsomething_new.so.1"]) == [
        "libsomething_new.so.1"
    ]


def test_the_hash_auditwheel_adds_does_not_orphan_a_library():
    """``libgfortran.so.5`` is grafted as ``libgfortran-a1b2c3d4.so.5``."""
    assert (
        notices.unclaimed_libraries(
            [
                "libgfortran-a1b2c3d4.so.5",
                "libquadmath-deadbeef.so.0",
                "libfmt-01234567.so.8",
                "libpugixml-89abcdef.so.1",
                "libspdlog-76543210.so.1",
            ]
        )
        == []
    )


def test_the_libraries_the_stack_actually_ships_are_all_claimed():
    """The DT_NEEDED closure of the payload, as the built prefix reports it."""
    shipped = [
        "libadios2_atl.so.2.12",
        "libadios2_core.so.2.12",
        "libadios2_core_mpi.so.2.12",
        "libadios2_cxx.so.2.12",
        "libadios2_cxx_mpi.so.2.12",
        "libadios2_dill.so.2.12",
        "libadios2_ffs.so.2.12",
        "libadios2_perfstubs.so.2.12",
        "libdolfinx.so.0.11",
        "libesmumps.so.7.0",
        "libhdf5.so.310",
        "libhdf5_hl.so.310",
        "libkahip.so",
        "libmetis.so",
        "libmpicxx.so.12",
        "libmpifort.so.12",
        "libopenblas.so.0",
        "libparhip_interface.so",
        "libpetsc.so.3.25",
        "libptesmumps.so.7.0",
        "libptscotch.so.7.0",
        "libptscotcherr.so.7.0",
        "libptscotchparmetisv3.so.7.0",
        "libscalapack.so.2.2",
        "libscotch.so.7.0",
        "libscotcherr.so.7.0",
        "libslepc.so.3.25",
        "libsuperlu_dist.so.9",
    ]

    assert notices.unclaimed_libraries(shipped) == []


def test_libmpi_is_claimed_by_nothing():
    """It is never in the wheel, so no entry may quietly account for it."""
    assert notices.unclaimed_libraries(["libmpi.so.12"]) == ["libmpi.so.12"]
