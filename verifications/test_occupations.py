"""The /occupations page and its read model (occupations_data.py)."""

import os

import pytest

import occupations_data
from occupations_data import load_occupations

HAVE_PROFILES = os.path.exists(occupations_data.PROFILES_FILE)


@pytest.mark.skipif(not HAVE_PROFILES, reason="data/processed/occupation_profiles.csv not built")
def test_profiles_load_with_both_measures():
    data = load_occupations()
    assert not data["failed"] and not data["missing"]
    assert data["summary"]["occupations"] == len(data["rows"]) > 0
    # The office-tool exclusion can only lower or keep the share, never raise it overall.
    assert data["summary"]["share_excl_office"] <= data["summary"]["share_all"]
    # Workers covered cannot exceed the BLS all-occupations total (about 155.5 million).
    assert data["summary"]["workers"] < 160_000_000


def test_a_missing_file_is_reported_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(occupations_data, "PROFILES_FILE", str(tmp_path / "absent.csv"))
    data = load_occupations()
    assert data["missing"] is True and data["rows"] == []


def test_the_page_is_public_and_renders():
    from fastapi.testclient import TestClient

    import auth
    import main

    assert auth.is_public("/occupations")
    response = TestClient(main.app).get("/occupations")
    assert response.status_code == 200
    assert "Occupations" in response.text
