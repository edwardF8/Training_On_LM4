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
