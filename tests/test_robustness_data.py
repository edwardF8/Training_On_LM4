"""Tests for data/robustness.py — manifest + restricted bio stream."""
import pytest

from data.bio_text import FIELD_SPECS
from data.robustness import (
    build_robustness_manifest,
    load_manifest,
    save_manifest,
)

FIELDS = ("birthday",)
N_TEMPLATES = len(FIELD_SPECS["birthday"]["templates"])  # 46


def test_manifest_counts_and_ranges():
    m = build_robustness_manifest(1000, FIELDS, limited_frac=0.2,
                                  template_frac=0.25, seed=0)
    assert len(m["limited_people"]) == 200          # round(0.2 * 1000)
    n_keep = round(0.25 * N_TEMPLATES)              # 12
    for idx, subs in m["limited_people"].items():
        assert 0 <= idx < 1000
        tlist = subs["birthday"]
        assert len(tlist) == n_keep
        assert len(set(tlist)) == n_keep            # no duplicate templates
        assert tlist == sorted(tlist)
        assert all(0 <= t < N_TEMPLATES for t in tlist)
    assert m["n_people"] == 1000
    assert m["fields"] == ["birthday"]
    assert m["n_templates"] == {"birthday": N_TEMPLATES}


def test_manifest_deterministic_and_seed_sensitive():
    a = build_robustness_manifest(500, FIELDS, seed=7)
    b = build_robustness_manifest(500, FIELDS, seed=7)
    c = build_robustness_manifest(500, FIELDS, seed=8)
    assert a == b
    assert a != c


def test_manifest_subsets_vary_across_people():
    m = build_robustness_manifest(1000, FIELDS, seed=0)
    subsets = {tuple(s["birthday"]) for s in m["limited_people"].values()}
    assert len(subsets) > 1


def test_manifest_round_trip_restores_int_keys(tmp_path):
    m = build_robustness_manifest(100, FIELDS, seed=3)
    p = tmp_path / "robustness_manifest.json"
    save_manifest(m, p)
    loaded = load_manifest(p)
    assert loaded == m
    assert all(isinstance(k, int) for k in loaded["limited_people"])
