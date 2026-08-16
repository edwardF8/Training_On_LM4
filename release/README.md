# release/ — shareable toy model + data seed-pack

See [`../AGENTS.md`](../AGENTS.md) for the full orientation.

| Path | What |
|---|---|
| `models/grid-L4-H6/` | BaselineL4 — 4 layers, 7.79M params, vocab 1836 |
| `models/grid-L8-H6/` | BaselineL8 — 8 layers, 14.87M params, vocab 1836 |
| `data/bioS_N-Bd_final_grid/` | dataset config, vocab remap, 50k people, robustness manifest |
| `regenerate_data.py` | rebuilds the two 180MB token files deterministically from seed |

The `.bin` token files are **not** in git (over GitHub's 100MB limit). Rebuild:

```bash
python release/regenerate_data.py          # full corpus + checksum verify
python release/regenerate_data.py --smoke  # seconds, no corpus written
```

Verified 2026-08-15: regenerating from seed reproduces all four artifacts
byte-for-byte, including both 180MB `.bin` files.
