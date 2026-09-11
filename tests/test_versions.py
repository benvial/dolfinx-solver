"""The package version and the DOLFINx release it ships, kept in step.

``__version__`` is the wheel's number and mirrors the upstream DOLFINx
release, with the ``.postN`` segment carrying packaging-only fixes.
``DOLFINX_VERSION`` is what the build scripts fetch from upstream, so it has
to stay a tag that upstream actually has.
"""

from wheelbuild.version import DOLFINX_VERSION, RELEASE, __version__


def test_the_package_and_upstream_versions_share_one_release():
    assert __version__.startswith(RELEASE)
    assert DOLFINX_VERSION.startswith(RELEASE)


def test_the_release_carries_no_packaging_segment():
    assert ".post" not in RELEASE


def test_packaging_fixes_only_ever_move_the_post_segment():
    """Anything else between the release and the post segment is a typo."""
    for version in (__version__, DOLFINX_VERSION):
        remainder = version.removeprefix(RELEASE)
        assert remainder == "" or remainder.startswith(".post")


def test_the_packaged_version_is_not_older_than_the_release_it_ships():
    """A wheel numbered below its upstream release would mislabel itself."""
    assert _post_number(__version__) >= _post_number(DOLFINX_VERSION)


def _post_number(version: str) -> int:
    _, _, post = version.partition(".post")
    return int(post) if post else -1
