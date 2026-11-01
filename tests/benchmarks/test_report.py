import csv
import json
from copy import deepcopy

import pytest

from benchmarks.report import load_campaign, ratio, summarize, write_timing_report


def record(version, mean, peak, status="ok"):
    return {
        "request": {
            "phase": "speed",
            "case": "ntxent",
            "profile": "standard",
            "dtype": "float32",
            "batch_size": 512,
            "version": version,
        },
        "status": status,
        "total_ms": {"mean": mean},
        "forward_ms": {"mean": mean * 0.6},
        "backward_ms": {"mean": mean * 0.4},
        "peak_allocated_bytes": peak,
        "peak_incremental_bytes": peak - 2**20,
        "peak_reserved_bytes": peak * 2,
        "loss": 7.0,
    }


def test_report_preserves_absolute_cost_process_range_and_worst_memory():
    rows = summarize(
        [
            record("3.0.0", 1, 3 * 2**20),
            record("3.0.0", 2, 4 * 2**20),
            record("3.0.0", 3, 5 * 2**20),
        ]
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["mean_ms"] == 2
    assert row["min_process_ms"] == 1
    assert row["max_process_ms"] == 3
    assert row["embeddings_per_second"] == 256000
    assert row["peak_incremental_mib"] == 4
    assert row["peak_allocated_mib"] == 5


def test_report_does_not_turn_partial_failures_into_timings_or_infinite_ratios():
    records = [
        record("2.9.0", 4, 9 * 2**20),
        record("2.9.0", 4, 9 * 2**20, "oom"),
        record("3.0.0", 2, 5 * 2**20),
    ]
    rows = summarize(records)
    assert rows[0]["status"] == "ok|oom"
    assert "mean_ms" not in rows[0]
    assert ratio("ntxent", {"rows": rows}, "mean_ms") == "ok|oom → ok"


def test_report_marks_overlapping_process_ranges():
    rows = summarize([record("2.9.0", 4, 9 * 2**20), record("3.0.0", 2, 5 * 2**20)])
    assert ratio("ntxent", {"rows": rows}, "mean_ms") == "2\N{MULTIPLICATION SIGN}"
    overlapping = deepcopy(rows)
    overlapping[0]["min_process_ms"] = 1.5
    assert ratio("ntxent", {"rows": overlapping}, "mean_ms").endswith("†")


def test_incomplete_matrix_cannot_be_published(tmp_path):
    config = {
        "cases": ["ntxent"],
        "profiles": {"standard": {}},
        "batch_sizes": [512],
        "bfloat16_batch_sizes": [],
        "versions": {"2.9.0": "a", "3.0.0": "b"},
        "rounds": 1,
        "seed": 17,
        "warmup": 1,
        "repeats": 1,
        "steps_per_block": 100,
    }
    (tmp_path / "campaign.json").write_text(json.dumps({"config": config}))
    (tmp_path / "results.jsonl").write_text("")
    with pytest.raises(ValueError, match="Incomplete speed matrix"):
        load_campaign(tmp_path)


def test_timing_only_export_keeps_protocol_and_requires_complete_timings(tmp_path):
    config = {
        "cases": ["ntxent"],
        "profiles": {
            "standard": {
                "dimensions": 128,
                "samples_per_class": 4,
                "proxy_classes": 1024,
            }
        },
        "batch_sizes": [512],
        "bfloat16_batch_sizes": [],
        "versions": {"2.9.0": "a", "3.0.0": "b"},
        "rounds": 1,
        "seed": 17,
        "warmup": 1,
        "repeats": 3,
        "steps_per_block": 10,
    }
    (tmp_path / "campaign.json").write_text(json.dumps({"config": config}))
    records = []
    for version in config["versions"]:
        r = record(version, 2, 4 * 2**20)
        r["key"] = f"speed-ntxent-standard-float32-512-{version}-0"
        r["environment"] = {
            "gpu": "NVIDIA Fixture",
            "gpu_total_bytes": 2**30,
            "cpu": "CPU",
            "pytorch": "2.14.0",
            "cuda_runtime": "13.0",
        }
        records.append(r)
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(FileNotFoundError):
        load_campaign(tmp_path)
    campaign = load_campaign(tmp_path, timing_only=True)
    output = tmp_path / "report"
    write_timing_report([campaign], output)
    with (output / "measurements.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(r["version"], r["mean_ms"]) for r in rows] == [
        ("2.9.0", "2"),
        ("3.0.0", "2"),
    ]
    assert "3 blocks of 10" in (output / "README.md").read_text()
    evidence = json.loads((output / "gpu-1-provenance.json").read_text())
    assert evidence["checks"] == []
    assert evidence["capacity"] == []
    path.write_text(json.dumps(records[0]))
    with pytest.raises(ValueError, match="Incomplete speed matrix"):
        load_campaign(tmp_path, timing_only=True)
