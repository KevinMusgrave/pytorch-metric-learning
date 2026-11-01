import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .compare import request_key, speed_plan

NAMES = {
    "ntxent": "NT-Xent",
    "contrastive": "Contrastive",
    "triplet": "All-triplet",
    "batch-hard-triplet": "Batch-hard + Triplet",
    "arcface": "ArcFace",
    "supcon": "SupCon",
    "multi-similarity": "MultiSimilarity",
    "proxy-anchor": "ProxyAnchor",
}


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        r = record["request"]
        if r["phase"] == "speed":
            groups[
                tuple(
                    r[k] for k in ("case", "profile", "dtype", "batch_size", "version")
                )
            ].append(record)
    rows = []
    for (case, profile, dtype, batch, version), group in sorted(groups.items()):
        statuses = sorted({r["status"] for r in group})
        row = {
            "case": case,
            "profile": profile,
            "dtype": dtype,
            "batch_size": batch,
            "version": version,
            "status": "|".join(statuses),
            "processes": len(group),
        }
        if statuses == ["ok"]:
            means = [r["total_ms"]["mean"] for r in group]
            row.update(
                mean_ms=statistics.mean(means),
                min_process_ms=min(means),
                max_process_ms=max(means),
            )
            row["embeddings_per_second"] = batch * 1000 / row["mean_ms"]
            for key in ("forward_ms", "backward_ms"):
                row[key] = statistics.mean(r[key]["mean"] for r in group)
            for key in (
                "peak_allocated_bytes",
                "peak_incremental_bytes",
                "peak_reserved_bytes",
            ):
                row[key.replace("bytes", "mib")] = max(r[key] for r in group) / 2**20
            row["loss_min"], row["loss_max"] = (
                min(r["loss"] for r in group),
                max(r["loss"] for r in group),
            )
        rows.append(row)
    return rows


def load_campaign(folder, timing_only=False):
    manifest = json.loads((folder / "campaign.json").read_text())
    config = manifest["config"]
    records = [
        json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()
    ]
    expected = {request_key(r) for r in speed_plan(config)}
    actual = {r["key"] for r in records if r["request"]["phase"] == "speed"}
    if actual != expected:
        raise ValueError(
            f"Incomplete speed matrix in {folder}: missing {len(expected - actual)}"
        )
    checks, capacity = [], []
    if not timing_only:
        checks = json.loads((folder / "correctness.json").read_text())
        if {r["case"] for r in checks if r["status"] == "passed"} != set(
            config["cases"]
        ):
            raise ValueError(f"Incomplete correctness evidence in {folder}")
        capacity = json.loads((folder / "capacity.json").read_text())
        if {(r["case"], r["version"]) for r in capacity} != {
            (c, v) for c in config["cases"] for v in config["versions"]
        }:
            raise ValueError(f"Incomplete capacity search in {folder}")
    environment = next(r["environment"] for r in records if "environment" in r)
    gpu = environment["gpu"].removeprefix("NVIDIA ").removeprefix("GeForce ")
    return {
        "manifest": manifest,
        "records": records,
        "checks": checks,
        "capacity": capacity,
        "environment": environment,
        "gpu": gpu,
        "rows": summarize(records),
    }


def number(value):
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.3g}"


def table(cases, campaigns, cell):
    lines = [
        "| Loss | " + " | ".join(c["gpu"] for c in campaigns) + " |",
        "| --- |" + " ---: |" * len(campaigns),
    ]
    lines.extend(
        "| " + NAMES[case] + " | " + " | ".join(cell(case, c) for c in campaigns) + " |"
        for case in cases
    )
    return "\n".join(lines)


def hardware_table(campaigns):
    lines = [
        "| GPU | GPU memory (GiB) | Host CPU | PyTorch | CUDA |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for campaign in campaigns:
        e = campaign["environment"]
        lines.append(
            f"| {campaign['gpu']} | {e['gpu_total_bytes'] / 2**30:.2f} | "
            f"{e['cpu']} | {e['pytorch']} | {e['cuda_runtime']} |"
        )
    return "\n".join(lines)


def selected(
    campaign, case, version="3.0.0", dtype="float32", profile="standard", batch=512
):
    return next(
        r
        for r in campaign["rows"]
        if (r["case"], r["version"], r["dtype"], r["profile"], r["batch_size"])
        == (case, version, dtype, profile, batch)
    )


def absolute(case, campaign, metric, dtype="float32", batch=512):
    row = selected(campaign, case, dtype=dtype, batch=batch)
    return number(row[metric]) if row["status"] == "ok" else row["status"]


def ratio(case, campaign, metric, profile="standard", dtype="float32", batch=512):
    old, new = (
        selected(campaign, case, version=v, profile=profile, dtype=dtype, batch=batch)
        for v in ("2.9.0", "3.0.0")
    )
    if old["status"] != "ok" or new["status"] != "ok":
        return f"{old['status']} → {new['status']}"
    value = number(old[metric] / new[metric]) + "\N{MULTIPLICATION SIGN}"
    if metric == "mean_ms" and max(old["min_process_ms"], new["min_process_ms"]) <= min(
        old["max_process_ms"], new["max_process_ms"]
    ):
        value += "†"
    return value


def capacity_cell(case, campaign):
    values = []
    for version in ("2.9.0", "3.0.0"):
        r = next(
            r
            for r in campaign["capacity"]
            if r["case"] == case and r["version"] == version
        )
        value = r["largest_verified_batch"]
        values.append(
            "unstable"
            if value is None
            else ("≥" if r["limiting_status"] == "search_cap" else "") + f"{value:,}"
        )
    return " → ".join(values)


def figures(campaign, output, slug):
    import pygal
    from pygal.style import Style

    style = Style(
        background="white",
        plot_background="white",
        foreground="#273343",
        foreground_strong="#142031",
        foreground_subtle="#dce1e7",
        guide_stroke_color="#e7ebef",
        major_guide_stroke_color="#cbd3dc",
        guide_stroke_dasharray="0",
        major_guide_stroke_dasharray="0",
        font_family="sans-serif",
        colors=(
            "#ec512c",
            "#2878b5",
            "#32936f",
            "#9b59b6",
            "#b07d12",
            "#cc497a",
            "#48795b",
            "#738096",
        ),
    )
    files = []
    positions = sorted({r["batch_size"] for r in campaign["rows"]})
    for metric, label in (
        ("mean_ms", "Forward + backward (ms, log scale)"),
        ("peak_incremental_mib", "Peak extra memory (MiB, log scale)"),
    ):
        chart = pygal.XY(
            style=style,
            width=1000,
            height=560,
            logarithmic=True,
            x_title="Batch size (log scale; 4 per class, or 1,024 proxy classes)",
            x_labels=positions,
            xrange=(positions[0] / 1.25, positions[-1] * 1.25),
            x_value_formatter=lambda x: f"{x:,.0f}",
            y_title=label,
            title=f"PML v3.0.0 · {campaign['gpu']} · float32 · 128 dimensions",
            legend_at_bottom=True,
            legend_at_bottom_columns=4,
            show_dots=True,
            js=[],
            print_values=False,
        )
        for case, name in NAMES.items():
            rows = [
                r
                for r in campaign["rows"]
                if r["case"] == case
                and r["profile"] == "standard"
                and r["dtype"] == "float32"
                and r["version"] == "3.0.0"
                and r["status"] == "ok"
            ]
            chart.add(name, [(r["batch_size"], r[metric]) for r in rows])
        filename = f"{slug}-{metric}.svg"
        (output / filename).write_bytes(chart.render() + b"\n")
        files.append(filename)
    return files


def write_artifacts(campaigns, output):
    output.mkdir(parents=True, exist_ok=True)
    config = campaigns[0]["manifest"]["config"]
    if any(c["manifest"]["config"] != config for c in campaigns):
        raise ValueError("GPU campaigns must use the same configuration")
    all_rows = []
    for index, campaign in enumerate(campaigns):
        slug = f"gpu-{index + 1}"
        for row in campaign["rows"]:
            all_rows.append({"gpu": campaign["gpu"], **row})
        with gzip.open(output / f"{slug}-raw.jsonl.gz", "wt") as stream:
            stream.writelines(json.dumps(r) + "\n" for r in campaign["records"])
        detail = {
            k: campaign[k]
            for k in ("gpu", "environment", "manifest", "checks", "capacity")
        }
        (output / f"{slug}-provenance.json").write_text(
            json.dumps(detail, indent=2) + "\n"
        )
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


def write_report(campaigns, output):
    write_artifacts(campaigns, output)
    config = campaigns[0]["manifest"]["config"]
    artifacts = []
    for index, campaign in enumerate(campaigns):
        slug = f"gpu-{index + 1}"
        artifacts.append((slug, figures(campaign, output, slug)))
    sections = [
        "# PML 3.0.0 GPU benchmarks",
        (
            "Released v3.0.0 and v2.9.0, measured on the same GPU and software "
            "stack. These are loss/miner forward and backward measurements; "
            "encoder, optimizer, data loading, compilation, anchor batching, "
            "and distributed training are excluded."
        ),
        (
            "All overview tables use float32, batch 512, and 128-dimensional "
            "seeded standard-normal embeddings. Pair/triplet objectives use "
            "four samples per class. "
            "ArcFace and ProxyAnchor have 1,024 fixed proxy classes and cyclic "
            "class labels. [Full measurements](measurements.csv) include the "
            "512-dimensional, sixteen-per-class profile with 10,000 proxy "
            "classes, all batch sizes, BF16, and process ranges."
        ),
        "## Hardware and software",
        hardware_table(campaigns),
        (
            "Memory is the total exposed by PyTorch, before the 90% capacity "
            "budget. Small eager workloads also depend on the host CPU; these "
            "results compare complete rented systems, not GPU silicon alone. "
            "Per-GPU provenance below includes Python, driver, power limits, "
            "clocks, timestamps, and precision settings."
        ),
        "## Version 3 speed",
        (
            "Milliseconds per forward/backward pair. Three process means, each "
            "averaging three blocks of 100 pairs after warmup; synchronization "
            "occurs only at block boundaries. Lower is better."
        ),
        table(config["cases"], campaigns, lambda case, c: absolute(case, c, "mean_ms")),
        "## Version 3 memory",
        (
            "Steady-state peak PyTorch allocation in MiB, including inputs, "
            "parameters, retained workspaces, and temporary tensors. Lower is "
            "better. Incremental allocation and allocator reservation are "
            "separate columns in the CSV; this is not total device usage."
        ),
        table(
            config["cases"],
            campaigns,
            lambda case, c: absolute(case, c, "peak_allocated_mib"),
        ),
        "## Batch capacity: v2.9 → v3.0",
        (
            "Largest verified batch under a 90% GPU tensor-allocation budget, "
            "with 128-dimensional float32 embeddings and the standard label/proxy "
            "configuration, using ordinary full loss calls and backward passes. "
            "Each successful boundary passed three fresh "
            "processes. Searches stop within 128 rows or 5% of a failing batch. ≥ "
            "marks the search ceiling, not a hardware limit. Per-GPU provenance "
            "files retain the failing batch and distinguish OOM, indexing, "
            "nonfinite, and timeout limits. This is loss-only capacity; a "
            "training model needs additional memory."
        ),
        table(config["cases"], campaigns, capacity_cell),
        "## Time ratio: v2.9 / v3.0",
        (
            "Above 1 means v3 is faster; below 1 means slower. † means the ranges "
            "of the three process means overlap, so the measured difference is "
            "not treated as established."
        ),
        table(config["cases"], campaigns, lambda case, c: ratio(case, c, "mean_ms")),
        "## Memory ratio: v2.9 / v3.0",
        "Ratio of peak PyTorch allocation. Above 1 means v3 uses less memory.",
        table(
            config["cases"],
            campaigns,
            lambda case, c: ratio(case, c, "peak_allocated_mib"),
        ),
        "## Larger embeddings and more positives",
        (
            "Time ratios at batch 512, 512 dimensions, sixteen samples per class "
            "(10,000 classes for proxy losses). Failed v2 runs are shown "
            "explicitly rather than assigned an infinite speedup."
        ),
        table(
            config["cases"],
            campaigns,
            lambda case, c: ratio(case, c, "mean_ms", "wide-dense"),
        ),
        "## Version 3 BF16",
        (
            "Milliseconds per pair at batch 512 in the standard profile. Inputs "
            "and parameters are BF16 directly; this is not an autocast or "
            "end-to-end mixed-precision training benchmark. Finite checks are "
            "required, but the cross-version full-gradient agreement check is "
            "float32."
        ),
        table(
            config["cases"],
            campaigns,
            lambda case, c: absolute(case, c, "mean_ms", "bfloat16"),
        ),
        "### BF16 at batch 2,048",
        (
            "Released v3.0.0 has major BF16 time and memory regressions for "
            "Contrastive and batch-hard triplet at this size. Their "
            "low-precision Euclidean distance uses the direct CUDA kernel to "
            "preserve nearby-point accuracy; its backward pass allocates a "
            "batch-by-batch-by-dimension buffer. These release measurements "
            "retain that behavior. See the [subsequent fix experiment]"
            "(../../performance.md#low-precision-distance-fix) for a separately "
            "measured correction included in v3.0.1."
        ),
        "Milliseconds per forward/backward pair:",
        table(
            config["cases"],
            campaigns,
            lambda case, c: absolute(case, c, "mean_ms", "bfloat16", 2048),
        ),
        "Peak PyTorch allocation in MiB:",
        table(
            config["cases"],
            campaigns,
            lambda case, c: absolute(case, c, "peak_allocated_mib", "bfloat16", 2048),
        ),
        "Time ratio, v2.9 / v3.0; below 1 means v3 is slower:",
        table(
            config["cases"],
            campaigns,
            lambda case, c: ratio(case, c, "mean_ms", dtype="bfloat16", batch=2048),
        ),
        "## Scaling and raw evidence",
        (
            "The charts show successful v3 float32 points in the standard "
            "profile. Memory charts show additional peak allocation above the "
            "warmed pre-forward allocation, excluding inputs, parameters, and "
            "workspaces already resident at that point. A missing point is a "
            "failed run, not zero cost; consult the "
            "CSV for its status. GPU and host CPU both affect small eager "
            "workloads. These rented machines were not isolated from host "
            "contention; process ranges quantify observed variation, not a "
            "universal performance guarantee."
        ),
    ]
    for c, (slug, charts) in zip(campaigns, artifacts, strict=True):
        sections.extend(
            [
                f"### {c['gpu']}",
                f"[Raw measurements]({slug}-raw.jsonl.gz) · "
                f"[Hardware and validation evidence]({slug}-provenance.json)",
                *(f"![{c['gpu']} scaling]({chart})" for chart in charts),
            ]
        )
    sections.extend(
        [
            "## Reproduce",
            (
                "See [the benchmark "
                "protocol](../../performance.md#systematic-release-comparison) and "
                "the checked-in `benchmarks/release.json` matrix. Raw evidence "
                "records the exact library commits, benchmark commit, GPU, driver, "
                "Python/PyTorch/CUDA versions, inputs, settings, and block averages."
            ),
        ]
    )
    (output / "README.md").write_text("\n\n".join(sections) + "\n")


def write_timing_report(campaigns, output):
    write_artifacts(campaigns, output)
    config = campaigns[0]["manifest"]["config"]
    sections = [
        "# Additional GPU timing measurements",
        f"{config['rounds']} fresh processes per configuration, each averaging "
        f"{config['repeats']} blocks of {config['steps_per_block']} forward/backward "
        "pairs after warmup. Times are milliseconds per pair; peak memory is "
        "total PyTorch allocation in MiB. Encoder, optimizer, and data loading "
        "are excluded. This campaign measures timing and memory; full-gradient "
        "agreement and capacity require separate evidence.",
        hardware_table(campaigns),
        "[CSV measurements](measurements.csv) retain full precision and statuses.",
    ]
    for name, profile in config["profiles"].items():
        sections.append(
            f"{name}: {profile['dimensions']} dimensions, "
            f"{profile['samples_per_class']} samples per class for pair/triplet "
            f"losses, {profile['proxy_classes']:,} classes for proxy losses."
        )
    keys = ("case", "profile", "dtype", "batch_size")
    groups = sorted({tuple(r[k] for k in keys) for r in campaigns[0]["rows"]})
    for group in groups:
        case, profile, dtype, batch = group
        lines = [
            "| GPU | Version | Status | Total | Forward | Backward | "
            "Process range | Peak MiB |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
        ]
        for campaign in campaigns:
            for row in campaign["rows"]:
                if tuple(row[k] for k in keys) != group:
                    continue
                ok = row["status"] == "ok"
                values = [
                    number(row[k]) if ok else "—"
                    for k in (
                        "mean_ms",
                        "forward_ms",
                        "backward_ms",
                        "peak_allocated_mib",
                    )
                ]
                span = (
                    f"{number(row['min_process_ms'])}\N{EN DASH}"
                    f"{number(row['max_process_ms'])}"
                    if ok
                    else "—"
                )
                cells = [
                    campaign["gpu"],
                    row["version"],
                    row["status"],
                    *values[:3],
                    span,
                    values[3],
                ]
                lines.append("| " + " | ".join(cells) + " |")
        sections.extend(
            [
                f"## {NAMES[case]} · {profile} · {dtype} · batch {batch:,}",
                "\n".join(lines),
            ]
        )
    sections.append("## Raw evidence")
    for index, campaign in enumerate(campaigns, 1):
        sections.append(
            f"{campaign['gpu']}: [raw records](gpu-{index}-raw.jsonl.gz) · "
            f"[configuration and hardware](gpu-{index}-provenance.json)"
        )
    (output / "README.md").write_text("\n\n".join(sections) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaigns", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    writer = write_timing_report if args.timing_only else write_report
    writer(
        [load_campaign(path, args.timing_only) for path in args.campaigns], args.output
    )


if __name__ == "__main__":
    main()
