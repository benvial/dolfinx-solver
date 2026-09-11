from wheelbuild import tag_check


def test_a_tag_naming_the_packaged_version_is_accepted():
    assert tag_check.tag_problem("v0.11.0.post0", version="0.11.0.post0") is None


def test_a_tag_from_another_version_is_reported():
    problem = tag_check.tag_problem("v0.12.0", version="0.11.0.post0")

    assert problem is not None
    assert "0.12.0" in problem
    assert "0.11.0.post0" in problem


def test_a_packaging_bump_the_tag_did_not_follow_is_reported():
    assert tag_check.tag_problem("v0.11.0.post1", version="0.11.0.post0") is not None


def test_a_tag_without_the_prefix_is_reported():
    problem = tag_check.tag_problem("0.11.0.post0", version="0.11.0.post0")

    assert problem is not None
    assert "v0.11.0.post0" in problem


def test_equivalent_spellings_of_one_version_are_still_two_tags():
    # Python's version rules read these as the same release; git does not, and
    # a release reachable under two tags is what this rejects.
    assert tag_check.tag_problem("v0.11.0.post01", version="0.11.0.post1") is not None


def test_the_command_line_reports_a_mismatch_as_a_failure(capsys, monkeypatch):
    monkeypatch.setattr(tag_check, "__version__", "0.11.0.post0")

    assert tag_check.main(["v9.9.9"]) == 1
    assert "9.9.9" in capsys.readouterr().err


def test_the_command_line_accepts_the_matching_tag(capsys, monkeypatch):
    monkeypatch.setattr(tag_check, "__version__", "0.11.0.post0")

    assert tag_check.main(["v0.11.0.post0"]) == 0
    assert "0.11.0.post0" in capsys.readouterr().out
