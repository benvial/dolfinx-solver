"""Keeping the repository's own dolfinx_solver out of the installed wheel's way."""

from pathlib import Path

from wheeltest import payload


def test_the_repository_itself_is_a_repository_entry():
    assert payload.repository_entry(str(payload.REPO_ROOT))


def test_the_working_directory_is_one_when_it_is_the_repository(monkeypatch):
    """`python -m` puts the working directory first, and it is usually a checkout."""
    monkeypatch.chdir(payload.REPO_ROOT)

    assert payload.repository_entry("")


def test_the_working_directory_is_not_one_anywhere_else(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert not payload.repository_entry("")


def test_the_installed_site_stays_on_the_path():
    entry = "/venv/lib/python3.12/site-packages"

    assert not payload.repository_entry(entry)
    assert entry in payload.installed_only([str(payload.REPO_ROOT), entry])


def test_the_repository_comes_off_however_it_is_spelled(monkeypatch):
    monkeypatch.chdir(payload.REPO_ROOT)
    path = [
        "",
        str(payload.REPO_ROOT),
        f"{payload.REPO_ROOT}/",
        "/venv/lib/python3.12/site-packages",
    ]

    assert payload.installed_only(path) == ["/venv/lib/python3.12/site-packages"]


def test_a_repository_elsewhere_is_left_alone(tmp_path):
    other = tmp_path / "another-checkout"
    other.mkdir()

    assert payload.installed_only([str(other)], repo_root=Path("/repo")) == [str(other)]
