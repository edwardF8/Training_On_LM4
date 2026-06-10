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


# ---------------------------------------------------------------------------
# Restricted bio stream
# ---------------------------------------------------------------------------

from collections import Counter

from data.bio_text import bio_stream, render_bio
from data.robustness import (
    render_bio_limited,
    robust_bio_stream,
    verify_limited_rendering,
)
from data.sample_people import sample_people

N_SMALL, K_SMALL = 30, 10


@pytest.fixture(scope="module")
def small_world():
    people = sample_people(N=N_SMALL, seed=0)
    manifest = build_robustness_manifest(N_SMALL, FIELDS, limited_frac=0.2,
                                         template_frac=0.25, seed=0)
    return people, manifest


def test_stream_pair_order_matches_bio_stream(small_world):
    people, manifest = small_world
    plain = [(i, e) for i, e, _ in
             bio_stream(people, K=K_SMALL, shuffle_seed=1, fields=FIELDS)]
    robust = [(i, e) for i, e, _ in
              robust_bio_stream(people, K=K_SMALL, manifest=manifest,
                                shuffle_seed=1, fields=FIELDS)]
    assert plain == robust


def test_stream_full_people_identical_to_bio_stream(small_world):
    people, manifest = small_world
    limited = set(manifest["limited_people"])
    plain = list(bio_stream(people, K=K_SMALL, shuffle_seed=1, fields=FIELDS))
    robust = list(robust_bio_stream(people, K=K_SMALL, manifest=manifest,
                                    shuffle_seed=1, fields=FIELDS))
    checked = 0
    for (i1, e1, t1), (i2, e2, t2) in zip(plain, robust):
        if i1 not in limited:
            assert t1 == t2
            checked += 1
    assert checked == (N_SMALL - len(limited)) * K_SMALL


def test_stream_every_person_emits_exactly_K(small_world):
    people, manifest = small_world
    counts = Counter(i for i, _, _ in
                     robust_bio_stream(people, K=K_SMALL, manifest=manifest,
                                       shuffle_seed=1, fields=FIELDS))
    assert len(counts) == N_SMALL
    assert all(c == K_SMALL for c in counts.values())
    assert sum(counts.values()) == N_SMALL * K_SMALL


def test_stream_limited_people_use_only_allowed_templates(small_world):
    people, manifest = small_world
    emitted_by_person = {}
    for i, _, text in robust_bio_stream(people, K=K_SMALL, manifest=manifest,
                                        shuffle_seed=1, fields=FIELDS):
        emitted_by_person.setdefault(i, []).append(text)

    for idx, subs in manifest["limited_people"].items():
        allowed = set(subs["birthday"])
        person = people[idx]
        renders = {t: render_bio_limited(person, {"birthday": t}, FIELDS)
                   for t in range(N_TEMPLATES)}
        for text in emitted_by_person[idx]:
            matches = {t for t, r in renders.items() if r == text}
            assert matches, f"person {idx}: emitted bio matches no template"
            assert matches <= allowed, (
                f"person {idx}: bio from disallowed template(s) {matches - allowed}")


def test_render_bio_limited_matches_render_bio_for_same_template(small_world):
    people, _ = small_world
    person = people[0]
    for t in (0, 7, 45):
        # render_bio picks template exposure_idx % 46, so exposure_idx=t
        # selects template t for the single birthday field.
        assert (render_bio_limited(person, {"birthday": t}, FIELDS)
                == render_bio(person, exposure_idx=t, fields=FIELDS))


def test_verify_limited_rendering_passes(small_world):
    people, manifest = small_world
    assert verify_limited_rendering(people, manifest, FIELDS, K=K_SMALL,
                                    n_check=3)
