"""The rules every cache marker in this build is read and written by.

Three stages leave a small file behind to tell the next run what an expensive
directory already holds: the install prefix's scalar variant, the rejection
recorded beside a refused download, and the digest an extracted source tree
was unpacked from. Each of them is written by a run that can be killed and
read by a run that has no other evidence, so the two rules below are what
keep a marker from turning an interrupted build into one a human has to
repair.
"""

import pytest

from wheelbuild import markers


def test_a_marker_that_is_not_there_says_nothing(tmp_path):
    assert markers.text(tmp_path / "absent") is None


def test_a_marker_is_read_without_its_trailing_newline(tmp_path):
    marker = tmp_path / "marker"
    marker.write_text("complex\n", encoding="utf-8")

    assert markers.text(marker) == "complex"


def test_a_marker_that_cannot_be_decoded_says_nothing(tmp_path):
    """A run killed mid-write is exactly how such a file appears, and every
    later run would otherwise raise on it rather than redo the work."""
    marker = tmp_path / "marker"
    marker.write_bytes(b"\xff\xfe")

    assert markers.text(marker) is None


def test_a_marker_that_is_a_directory_says_nothing(tmp_path):
    (tmp_path / "marker").mkdir()

    assert markers.text(tmp_path / "marker") is None


def test_writing_a_marker_leaves_the_text_it_was_given(tmp_path):
    """The terminator is the writer's, so every marker ends the same way."""
    marker = tmp_path / "marker"

    markers.write(marker, "real")

    assert marker.read_text(encoding="utf-8") == "real\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["marker"]


def test_what_was_written_is_what_is_read_back(tmp_path):
    """A marker holding two lines is the rejection one, which is compared
    field by field: the round trip has to keep both."""
    marker = tmp_path / "marker"

    markers.write(marker, "the digest wanted\nthe digest that arrived")

    assert markers.text(marker) == "the digest wanted\nthe digest that arrived"


def test_an_interrupted_write_leaves_the_previous_marker_whole(tmp_path):
    """Writing in place truncates first, so a run killed in the window leaves
    an empty file -- the state ticket 24 is about. The new text lands beside
    the marker and is renamed over it, which within one directory either
    happens or does not, so a write that fails leaves the old answer and no
    temporary in a cache that is kept between runs."""
    marker = tmp_path / "marker"
    marker.write_text("the previous claim\n", encoding="utf-8")

    with pytest.raises(UnicodeEncodeError):
        # A lone surrogate fails while the bytes are being produced, which is
        # the closest a test gets to a process killed between two writes.
        markers.write(marker, "\ud800")

    assert marker.read_text(encoding="utf-8") == "the previous claim\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["marker"]
