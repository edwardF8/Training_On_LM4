"""Bucket-math tests for eval/robustness_probe.py.

score_pairs_fp_lp_batched (the batched decoder) is stubbed — no model, no
tokenizer, no GPU. The stub makes FP correct exactly on (limited person, seen
template) pairs and LP correct everywhere, so every bucket has a known
expected value. (Equality of the real batched decoder with the per-pair
score_pair lives in tests/test_probe_batching.py.)
"""
import math

from data.bio_text import FIELD_SPECS
from data.robustness import build_robustness_manifest
from data.sample_people import sample_people
import eval.robustness_probe as rp

FIELDS = ("birthday",)
TEMPLATES = FIELD_SPECS["birthday"]["templates"]
N_T = len(TEMPLATES)
BUCKETS = ("total", "limitedSet", "fullSet", "limitedSeen", "limitedUnseen")


class FakeModel:
    def eval(self):
        pass


class FakeTokenizer:
    eos_token_id = 50256


def test_select_eval_people_first_k_per_group():
    manifest = build_robustness_manifest(200, FIELDS, limited_frac=0.2, seed=0)
    limited, full = rp.select_eval_people(manifest, 200, max_people_per_group=10)
    lim_all = set(manifest["limited_people"])
    assert limited == sorted(lim_all)[:10]
    assert full == [i for i in range(200) if i not in lim_all][:10]
    assert set(limited).isdisjoint(full)


def test_run_probe_bucket_math(monkeypatch):
    n_people, per_group = 40, 4
    people = sample_people(N=n_people, seed=0)
    manifest = build_robustness_manifest(n_people, FIELDS, limited_frac=0.2,
                                         template_frac=0.25, seed=0)
    allowed_by_idx = {i: set(s["birthday"])
                      for i, s in manifest["limited_people"].items()}
    id_to_idx = {people[i]["id"]: i for i in range(n_people)}
    t_to_idx = {t: i for i, t in enumerate(TEMPLATES)}

    def fake_batched(model, tokenizer, old_to_new, new_to_old, eos_remapped,
                     pairs, device, batch_size=128):
        out = []
        for person, template in pairs:
            p_idx = id_to_idx[person["id"]]
            t_idx = t_to_idx[template]
            fp = int(p_idx in allowed_by_idx and t_idx in allowed_by_idx[p_idx])
            out.append({"FP": fp, "LP": 1})
        return out

    monkeypatch.setattr(rp, "score_pairs_fp_lp_batched", fake_batched)

    res = rp.run_probe(FakeModel(), FakeTokenizer(), {50256: 0}, people,
                       manifest, max_people_per_group=per_group, device="cpu")

    n_keep = round(0.25 * N_T)  # 12
    macro, counts = res["macro"], res["counts"]

    # attempted-pair counts per bucket
    assert counts["FP"]["total"][1] == 2 * per_group * N_T
    assert counts["FP"]["limitedSet"][1] == per_group * N_T
    assert counts["FP"]["fullSet"][1] == per_group * N_T
    assert counts["FP"]["limitedSeen"][1] == per_group * n_keep
    assert counts["FP"]["limitedUnseen"][1] == per_group * (N_T - n_keep)

    # FP was made correct exactly on limitedSeen pairs
    assert macro["FP"]["limitedSeen"] == 1.0
    assert macro["FP"]["limitedUnseen"] == 0.0
    assert macro["FP"]["fullSet"] == 0.0
    assert math.isclose(macro["FP"]["limitedSet"], n_keep / N_T)
    assert math.isclose(macro["FP"]["total"],
                        (per_group * n_keep) / (2 * per_group * N_T))

    # LP correct everywhere
    assert all(macro["LP"][b] == 1.0 for b in BUCKETS)

    # bookkeeping
    assert res["n_people"] == {"limited": per_group, "full": per_group}
    assert res["n_templates"] == N_T
    assert len(res["eval_people"]["limited"]) == per_group
    assert len(res["eval_people"]["full"]) == per_group
    assert set(res["per_template"]) == set(range(N_T))
    for d in res["per_template"].values():
        assert set(d) == {"FP_total", "FP_limited", "FP_full",
                          "LP_total", "LP_limited", "LP_full"}


def test_make_robust_runner_bakes_in_group_size(monkeypatch):
    manifest = build_robustness_manifest(40, FIELDS, seed=0)
    fake_results = {
        "macro": {m: {b: 0.5 for b in BUCKETS} for m in ("FP", "LP")},
        "counts": {},
        "per_template": {0: {"FP_total": 0.5, "FP_limited": 0.5, "FP_full": 0.5,
                             "LP_total": 0.5, "LP_limited": 0.5, "LP_full": 0.5}},
        "eval_people": {"limited": [], "full": []},
        "n_people": {"limited": 0, "full": 0},
        "n_templates": N_T,
    }
    captured = {}

    def fake_run_probe(model, tokenizer, old_to_new, people, mani, *,
                       max_people_per_group, device=None):
        captured["mppg"] = max_people_per_group
        return fake_results

    monkeypatch.setattr(rp, "run_probe", fake_run_probe)

    runner = rp.make_robust_runner(manifest, max_people_per_group=77)
    results, payload, tables, json_name = runner(None, None, {}, [], FIELDS, 50)

    assert captured["mppg"] == 77  # baked-in wins; registry's max_people=50 ignored
    assert json_name == "probe_robust.json"
    assert results is fake_results
    for m in ("FP", "LP"):
        for b in BUCKETS:
            assert payload[f"robustProbe/{m}/{b}"] == 0.5
    assert set(payload) == {f"robustProbe/{m}/{b}"
                            for m in ("FP", "LP") for b in BUCKETS}
    assert "robustProbe/per_template" in tables
