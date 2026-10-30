"""Tests never read or migrate real user project folders."""

import pytest


@pytest.fixture(autouse=True)
def isolated_projects_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("WEAVE_PROJECTS_DIR", str(tmp_path / "user-projects"))
