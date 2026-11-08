import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmarks.compare import Campaign, capacity_search, git_revision, speed_plan
from benchmarks.measurement import workload

ROOT = Path(__file__).resolve().parents[2]


def configuration():
    return json.loads((ROOT / "benchmarks/release.json").read_text())


def test_speed_matrix_preserves_paired_inputs_and_three_processes():
    config = configuration()
    plan = list(speed_plan(config))
    expected_groups = len(config["cases"]) * (
        len(config["profiles"]) * len(config["batch_sizes"])
        + len(config["bfloat16_batch_sizes"])
    )
    assert len(plan) == expected_groups * 2 * 3
    orders = set()
    for a, b in zip(plan[::2], plan[1::2], strict=True):
        orders.add((a["version"], b["version"]))
        assert {a["version"], b["version"]} == {"2.9.0", "3.0.0"}
        assert {k: v for k, v in a.items() if k != "version"} == {
            k: v for k, v in b.items() if k != "version"
        }
    assert len(orders) == 2


@pytest.mark.parametrize("failure", ["oom", "index_limit", "timeout", "nonfinite"])
def test_capacity_keeps_observed_failure_and_confirms_success(failure):
    calls = []

    def probe(batch, repetition):
        calls.append((batch, repetition))
        return {"status": "ok" if batch <= 1400 else failure}

    result = capacity_search(probe, configuration()["capacity"])
    assert result["largest_verified_batch"] == 1280
    assert result["first_failing_batch"] == 1408
    assert result["limiting_status"] == failure
    assert calls[-2:] == [(1280, 1), (1280, 2)]


@pytest.mark.parametrize(
    "statuses, expected_status",
    [(("oom", "ok"), "oom"), (("ok", "timeout"), "timeout")],
)
def test_capacity_reports_lower_bound_and_unstable_confirmation(
    statuses, expected_status
):
    settings = {**configuration()["capacity"], "maximum": 512}
    result = capacity_search(lambda batch, repetition: {"status": "ok"}, settings)
    assert result["largest_verified_batch"] == 512
    assert result["first_failing_batch"] is None
    assert result["limiting_status"] == "search_cap"
    result = capacity_search(
        lambda batch, repetition: {
            "status": "ok" if repetition == 0 else statuses[repetition - 1]
        },
        settings,
    )
    assert result["largest_verified_batch"] is None
    assert result["first_failing_batch"] is None
    assert result["limiting_status"] == expected_status
    assert result["confirmation_statuses"] == list(statuses)


def test_proxy_initialization_and_labels_are_independent_of_batch_composition():
    base = {
        "case": "arcface",
        "version": "3.0.0",
        "proxy_classes": 8,
        "dimensions": 4,
        "dtype": "float32",
        "device": "cpu",
        "seed": 17,
        "batch_size": 16,
        "samples_per_class": 4,
    }
    module, embeddings, labels = workload(base)
    other, repeated, changed_labels = workload({**base, "samples_per_class": 16})
    torch.testing.assert_close(embeddings, repeated, rtol=0, atol=0)
    torch.testing.assert_close(module.weight, other.weight, rtol=0, atol=0)
    assert labels.tolist() == list(range(8)) * 2
    assert torch.equal(labels, changed_labels)
    assert module.margin == 0.5


def test_real_worker_results_can_resume_but_not_mix_configuration(tmp_path):
    config = configuration()
    config["versions"]["3.0.0"] = git_revision(ROOT)
    campaign = Campaign(config, {"3.0.0": ROOT}, tmp_path, "cpu", False)
    request = {
        "phase": "speed",
        "case": "ntxent",
        "profile": "tiny",
        "batch_size": 8,
        "dtype": "float32",
        "version": "3.0.0",
        "round": 0,
        "dimensions": 4,
        "samples_per_class": 4,
        "proxy_classes": 8,
        "warmup": 1,
        "repeats": 2,
    }
    record = campaign.run(request)
    assert record["status"] == "ok"
    assert len(record["samples"]) == 2
    assert record["peak_allocated_bytes"] is None
    for sample in record["samples"]:
        assert sample["total_ms"] == pytest.approx(
            sample["elapsed_ms"] / sample["steps"]
        )
        assert sample["total_ms"] >= sample["forward_ms"] + sample["backward_ms"]
    resumed = Campaign(config, {"3.0.0": ROOT}, tmp_path, "cpu", True)
    assert resumed.run(request) == record
    assert len(campaign.results.read_text().splitlines()) == 1
    changed = {**config, "seed": 18}
    with pytest.raises(ValueError, match="Existing campaign differs"):
        Campaign(changed, {"3.0.0": ROOT}, tmp_path, "cpu", True)


def test_original_runner_executes_small_real_workloads(tmp_path):
    output = tmp_path / "smoke.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.run",
            "--batch-size",
            "8",
            "--dimensions",
            "4",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    rows = json.loads(output.read_text())["results_by_rank"][0]
    assert {"ntxent", "triplet", "anchor-batched-ntxent"} <= {r["case"] for r in rows}
    assert all(len(row["samples"]) == 2 for row in rows)


def test_capacity_no_verified_batch_and_unaligned_start():
    settings = {
        "start": 9,
        "maximum": 64,
        "granularity": 4,
        "relative_resolution": 0.01,
    }
    empty = capacity_search(lambda *_: {"status": "oom"}, settings)
    assert empty["largest_verified_batch"] is None
    assert empty["limiting_status"] == "oom"
    assert empty["confirmation_statuses"] == []
    result = capacity_search(
        lambda batch, _: {"status": "ok" if batch <= 16 else "oom"}, settings
    )
    assert result["largest_verified_batch"] == 16
    assert result["first_failing_batch"] == 18


@pytest.mark.parametrize(
    "invalid",
    [
        {"start": 0},
        {"start": 65},
        {"maximum": 8},
        {"granularity": 0},
        {"granularity": -1},
        {"relative_resolution": -0.1},
        {"relative_resolution": 1.0},
        {"relative_resolution": float("nan")},
    ],
)
def test_capacity_rejects_invalid_settings_before_running_a_probe(invalid):
    settings = {
        "start": 9,
        "maximum": 64,
        "granularity": 4,
        "relative_resolution": 0.01,
    }
    with pytest.raises(ValueError):
        capacity_search(
            lambda *_: pytest.fail("invalid settings must not launch a probe"),
            {**settings, **invalid},
        )
