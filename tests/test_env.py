"""Hibana's names first, then the former SHARAR_* ones and ~/.local/share/sharar."""
from evals import datasets
from evals.env import former, setting


def test_setting_prefers_hibana_then_former_name():
    assert setting("RESULTS", {"HIBANA_RESULTS": "/h", "SHARAR_RESULTS": "/s"}) == "/h"
    assert setting("RESULTS", {"SHARAR_RESULTS": "/s"}) == "/s"
    assert setting("RESULTS", {"HIBANA_RESULTS": "", "SHARAR_RESULTS": "/s"}) == "/s"  # empty counts as unset
    assert setting("RESULTS", {}) is None


def test_former_directory_only_while_the_new_one_is_missing(tmp_path):
    new, old = tmp_path / "hibana", tmp_path / "sharar"
    assert former(new, old) == new  # neither exists
    old.mkdir()
    assert former(new, old) == old
    new.mkdir()
    assert former(new, old) == new


def test_results_root(tmp_path, monkeypatch):
    for name in ("HIBANA_RESULTS", "SHARAR_RESULTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert datasets.results_root() == tmp_path / "hibana"
    (tmp_path / "sharar").mkdir()
    assert datasets.results_root() == tmp_path / "sharar"
    monkeypatch.setenv("SHARAR_RESULTS", str(tmp_path / "s"))
    assert datasets.results_root() == tmp_path / "s"
    monkeypatch.setenv("HIBANA_RESULTS", str(tmp_path / "h"))
    assert datasets.results_root() == tmp_path / "h"
