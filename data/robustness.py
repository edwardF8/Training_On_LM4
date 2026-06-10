# data/robustness.py
#
# RobustnessTest data generation — see
# docs/superpowers/specs/2026-06-10-robustness-test-design.md
#
# A seeded "manifest" marks a fraction of people (default 20%) as LIMITED:
# each limited person's bios are rendered from their own random subset of
# templates (default round(0.25*T), i.e. 12 of the 46 birthday templates)
# instead of the full round-robin. Every person still emits exactly K bios,
# so the corpus shape is unchanged; only the paraphrase diversity of limited
# people drops. The manifest is the on-disk record of who is limited and
# which template indices each kept.
#
# Pure add-on: data/bio_text.py is untouched. render_bio_limited mirrors
# render_bio's rendering loop with caller-chosen template indices.

import json
import random
from pathlib import Path

from data.bio_text import FIELD_SPECS, render_bio


def build_robustness_manifest(n_people, fields, limited_frac=0.2,
                              template_frac=0.25, seed=0):
    """Choose which people are limited and which templates each keeps.

    Deterministic in `seed` (one random.Random drives both draws; limited
    indices are processed in ascending order). Each limited person's subset is
    sampled independently, so subsets differ across people.

    Returns the manifest dict (JSON-serializable; person keys are ints
    in memory, strings on disk — load_manifest restores ints).
    """
    rng = random.Random(seed)
    n_limited = round(limited_frac * n_people)
    limited_indices = sorted(rng.sample(range(n_people), n_limited))

    limited_people = {}
    for idx in limited_indices:
        subsets = {}
        for field in fields:
            n_templates = len(FIELD_SPECS[field]["templates"])
            n_keep = max(1, round(template_frac * n_templates))
            subsets[field] = sorted(rng.sample(range(n_templates), n_keep))
        limited_people[idx] = subsets

    return {
        "seed": seed,
        "limited_frac": limited_frac,
        "template_frac": template_frac,
        "n_people": n_people,
        "fields": list(fields),
        "n_templates": {f: len(FIELD_SPECS[f]["templates"]) for f in fields},
        "limited_people": limited_people,
    }


def save_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path):
    """Load a manifest; person-index keys come back as ints."""
    with open(path) as f:
        manifest = json.load(f)
    manifest["limited_people"] = {
        int(k): v for k, v in manifest["limited_people"].items()
    }
    return manifest


def render_bio_limited(person, template_idx_by_field, fields):
    """Mirror of data.bio_text.render_bio with caller-chosen template indices.

    render_bio derives template_idx = exposure_idx % T per field; here the
    caller supplies the index for each field explicitly (RobustnessTest
    restricts limited people to a per-person template subset). Output is
    byte-identical to render_bio for the same (person, template) choice.
    No reverse_md support — the robustness pipeline doesn't use it.
    """
    name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
    pronoun = "He" if person["id"] % 2 == 0 else "She"

    parts = []
    for field in fields:
        spec = FIELD_SPECS[field]
        template = spec["templates"][template_idx_by_field[field]]
        subject = name if spec["subject"] == "name" else pronoun
        value = spec["render_value"](person)
        parts.append(" " + template.format(
            name=subject,
            **{spec["value_placeholder"]: value},
        ))
    return "".join(parts)


def robust_bio_stream(people, K, manifest, shuffle_seed=1, fields=("birthday",)):
    """Yield (person_idx, exposure_idx, bio_text) like bio_text.bio_stream.

    The shuffled pair sequence is IDENTICAL to bio_stream's for the same
    shuffle_seed (same pair construction, same Random), so the packed corpus
    differs from a normal run only in the text of limited people's bios.

    Full-set people render exactly as bio_stream does. Limited people map
    exposure e -> their allowed template allowed[e % len(allowed)] per field,
    so each still emits exactly K bios (each allowed template ~K/len(allowed)
    times).
    """
    limited = {int(i): subs for i, subs in manifest["limited_people"].items()}
    pairs = [(i, e) for i in range(len(people)) for e in range(K)]
    random.Random(shuffle_seed).shuffle(pairs)
    for i, e in pairs:
        subs = limited.get(i)
        if subs is None:
            yield i, e, render_bio(people[i], exposure_idx=e, fields=fields)
        else:
            tidx = {f: subs[f][e % len(subs[f])] for f in fields}
            yield i, e, render_bio_limited(people[i], tidx, fields)


def verify_limited_rendering(people, manifest, fields, K, n_check=5):
    """Seatbelt for the ablation script: re-render the first n_check limited
    people's K bios and assert each is the rendering of an allowed template —
    and not of any disallowed one. (Template strings are pairwise distinct, so
    a bio's text identifies its template.) Raises AssertionError on violation;
    returns True otherwise.
    """
    limited = sorted(int(i) for i in manifest["limited_people"])
    for idx in limited[:n_check]:
        person = people[idx]
        subs = {f: list(manifest["limited_people"][idx][f]) for f in fields}
        for e in range(K):
            tidx = {f: subs[f][e % len(subs[f])] for f in fields}
            text = render_bio_limited(person, tidx, fields)
            for f in fields:
                n_templates = len(FIELD_SPECS[f]["templates"])
                allowed = set(subs[f])
                hits = {t for t in range(n_templates)
                        if render_bio_limited(person, {**tidx, f: t}, fields)
                        == text}
                if not hits & allowed:
                    raise AssertionError(
                        f"person {idx}, exposure {e}: bio matches no allowed "
                        f"{f} template {sorted(allowed)}")
                if hits - allowed:
                    raise AssertionError(
                        f"person {idx}, exposure {e}: bio matches disallowed "
                        f"{f} template(s) {sorted(hits - allowed)}")
    return True
