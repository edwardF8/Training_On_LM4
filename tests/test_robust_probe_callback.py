"""CSV schema tests for RobustProbeAtEpochs (scheduling is inherited and
already exercised in production by ProbeAtEpochs; only _append_csv differs)."""
import csv

from eval.robust_probe_callback import ROBUST_CSV_COLUMNS, RobustProbeAtEpochs


def _make_cb(tmp_path):
    return RobustProbeAtEpochs(
        probes=("birthday_robust",), tokenizer=None, old_to_new={},
        people=[], fields=("birthday",), out_dir=str(tmp_path / "run"),
        run_name="grid-L4-H6", probe_epochs=[1, 2], max_epochs=2,
        csv_path=tmp_path / "probe_curve.csv",
        run_info={"family": "fam", "study": "grid", "numLayers": 4,
                  "numHeads": 6, "dmodel": 384},
    )


def test_append_csv_writes_all_robust_columns(tmp_path):
    cb = _make_cb(tmp_path)
    results = {"birthday_robust": {"macro": {
        "FP": {"total": 0.83, "limitedSet": 0.71, "fullSet": 0.95,
               "limitedSeen": 0.88, "limitedUnseen": 0.65},
        "LP": {"total": 0.9, "limitedSet": 0.8, "fullSet": 0.99,
               "limitedSeen": 0.91, "limitedUnseen": 0.75},
    }}}
    cb._append_csv(epoch=2, results=results)

    with open(tmp_path / "probe_curve.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert list(row.keys()) == ROBUST_CSV_COLUMNS
    assert row["family"] == "fam"
    assert row["run_name"] == "grid-L4-H6"
    assert row["epoch"] == "2"
    assert row["FP_total"] == "0.83"
    assert row["FP_limited"] == "0.71"
    assert row["FP_full"] == "0.95"
    assert row["FP_limitedSeen"] == "0.88"
    assert row["FP_limitedUnseen"] == "0.65"
    assert row["LP_total"] == "0.9"
    assert row["LP_limitedUnseen"] == "0.75"


def test_append_csv_blank_metrics_when_probe_results_missing(tmp_path):
    cb = _make_cb(tmp_path)
    cb._append_csv(epoch=1, results={})
    with open(tmp_path / "probe_curve.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["FP_total"] == ""
    assert rows[0]["family"] == "fam"
    assert rows[0]["epoch"] == "1"


def test_append_csv_appends_without_duplicate_header(tmp_path):
    cb = _make_cb(tmp_path)
    cb._append_csv(epoch=1, results={})
    cb._append_csv(epoch=2, results={})
    with open(tmp_path / "probe_curve.csv") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 3  # 1 header + 2 rows
