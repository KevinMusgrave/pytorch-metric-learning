import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def git_revision(root):
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def speed_plan(config):
    groups = [
        (case, profile, batch, "float32")
        for case in config["cases"]
        for profile in config["profiles"]
        for batch in config["batch_sizes"]
    ]
    groups.extend(
        (case, "standard", batch, "bfloat16")
        for case in config["cases"]
        for batch in config["bfloat16_batch_sizes"]
    )
    versions = list(config["versions"])
    for repetition in range(config["rounds"]):
        random.Random(config["seed"] + repetition).shuffle(groups)
        for index, (case, profile, batch, dtype) in enumerate(groups):
            order = versions if (index + repetition) % 2 == 0 else versions[::-1]
            for version in order:
                yield {
                    "phase": "speed",
                    "case": case,
                    "profile": profile,
                    "batch_size": batch,
                    "dtype": dtype,
                    "version": version,
                    "round": repetition,
                    **config["profiles"][profile],
                    "warmup": config["warmup"],
                    "repeats": config["repeats"],
                    "steps_per_block": config["steps_per_block"],
                }


def request_key(request):
    fields = ("phase", "case", "profile", "dtype", "batch_size", "version", "round")
    return "-".join(str(request[field]) for field in fields)


def capacity_search(probe, settings):
    low, high = 0, settings["start"]
    failure = None
    while high <= settings["maximum"]:
        result = probe(high, 0)
        if result["status"] != "ok":
            failure = result["status"]
            break
        low = high
        if high == settings["maximum"]:
            break
        high = min(high * 2, settings["maximum"])
    if failure:
        quantum = settings["granularity"]
        while high - low > max(quantum, low * settings["relative_resolution"]):
            middle = ((low + high) // (2 * quantum)) * quantum
            if middle <= low:
                break
            result = probe(middle, 0)
            if result["status"] == "ok":
                low = middle
            else:
                high, failure = middle, result["status"]
    confirmations = [probe(low, i)["status"] for i in (1, 2)] if low else []
    return {
        "largest_verified_batch": low
        if all(s == "ok" for s in confirmations)
        else None,
        "first_failing_batch": high if failure else None,
        "limiting_status": failure or "search_cap",
        "confirmation_statuses": confirmations,
    }


class Campaign:
    def __init__(self, config, sources, output, device, resume):
        self.config, self.sources, self.output, self.device = (
            config,
            sources,
            output,
            device,
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.work = self.output / "work"
        self.work.mkdir(exist_ok=True)
        self.records = {}
        manifest = {
            "config": config,
            "sources": {v: str(p.resolve()) for v, p in sources.items()},
            "runner_revision": git_revision(Path(__file__).resolve().parents[1]),
            "device": device,
        }
        if device == "cuda":
            manifest["gpu_identity"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        path = self.output / "campaign.json"
        if path.exists():
            if not resume or json.loads(path.read_text()) != manifest:
                raise ValueError(
                    "Existing campaign differs; use a new output directory"
                )
        else:
            path.write_text(json.dumps(manifest, indent=2) + "\n")
        self.results = self.output / "results.jsonl"
        if self.results.exists():
            for line in self.results.read_text().splitlines():
                record = json.loads(line)
                self.records[record["key"]] = record

    def run(self, spec):
        key = request_key(spec)
        if key in self.records:
            return self.records[key]
        request = {
            **spec,
            "source": str(self.sources[spec["version"]].resolve()),
            "revision": self.config["versions"][spec["version"]],
            "device": self.device,
            "seed": self.config["seed"],
            "threads": self.config["threads"],
            "memory_fraction": self.config["memory_fraction"],
            "steps_per_block": spec.get("steps_per_block", 1),
            "warmup_seconds": self.config["warmup_seconds"]
            if spec["phase"] == "speed"
            else 0,
            "gradient_file": str(self.work / f"{key}.pt"),
            "result_file": str(self.work / f"{key}.json"),
        }
        request_file = self.work / f"{key}.request.json"
        request_file.write_text(json.dumps(request))
        command = [sys.executable, "-m", "benchmarks.measurement", str(request_file)]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config["timeout_seconds"],
            )
        except subprocess.TimeoutExpired:
            record = {"status": "timeout", "request": request}
        else:
            if completed.returncode:
                (self.work / f"{key}.log").write_text(
                    completed.stdout + completed.stderr
                )
                raise RuntimeError(
                    f"Benchmark worker failed: {key}\n{completed.stderr[-4000:]}"
                )
            record = json.loads(Path(request["result_file"]).read_text())
        record["key"] = key
        with self.results.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
        self.records[key] = record
        timings = record.get("total_ms")
        timing = timings["mean"] if isinstance(timings, dict) else None
        print(
            f"{key}: {record['status']} {timing if timing is not None else ''}",
            flush=True,
        )
        return record

    def correctness(self):
        import torch

        checks = []
        for case in self.config["cases"]:
            records = [
                self.run(
                    {
                        "phase": "correctness",
                        "case": case,
                        "profile": "reference",
                        "batch_size": 64,
                        "dtype": "float32",
                        "version": version,
                        "round": 0,
                        "dimensions": 32,
                        "samples_per_class": 4,
                        "proxy_classes": 32,
                        "warmup": 0,
                        "repeats": 1,
                    }
                )
                for version in self.config["versions"]
            ]
            if any(record["status"] != "ok" for record in records):
                raise RuntimeError(f"Correctness workload failed: {case}")
            tensors = [
                torch.load(r["request"]["gradient_file"], weights_only=True)
                for r in records
            ]
            errors = {}
            for name in tensors[0]:
                a, b = (t[name].float() for t in tensors)
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)
                errors[name] = {"max_absolute_difference": (a - b).abs().max().item()}
            checks.append({"case": case, "status": "passed", "comparisons": errors})
        (self.output / "correctness.json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )

    def capacity(self):
        settings = self.config["capacity"]
        profile = settings["profile"]
        results = []
        for case in self.config["cases"]:
            for version in self.config["versions"]:

                def probe(batch, repetition, case=case, version=version):
                    return self.run(
                        {
                            "phase": "capacity",
                            "case": case,
                            "profile": profile,
                            "batch_size": batch,
                            "dtype": "float32",
                            "version": version,
                            "round": repetition,
                            **self.config["profiles"][profile],
                            "warmup": settings["warmup"],
                            "repeats": settings["repeats"],
                        }
                    )

                result = capacity_search(probe, settings)
                results.append(
                    {"case": case, "version": version, "profile": profile, **result}
                )
                (self.output / "capacity.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )


def main():
    runner_root = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(
        ["git", "-C", str(runner_root), "status", "--porcelain", "--", "benchmarks"],
        text=True,
    )
    if dirty:
        raise ValueError("Commit benchmark source before recording a campaign")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("benchmarks/release.json"))
    parser.add_argument("--source-2", type=Path, required=True)
    parser.add_argument("--source-3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("correctness", "speed", "capacity"),
        default=["correctness", "speed", "capacity"],
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sources = {"2.9.0": args.source_2, "3.0.0": args.source_3}
    if set(config["versions"]) != set(sources):
        parser.error("The comparison supports exactly versions 2.9.0 and 3.0.0")
    for version, source in sources.items():
        if git_revision(source) != config["versions"][version]:
            parser.error(f"Incorrect source revision for {version}")
    campaign = Campaign(
        config, sources, args.output.resolve(), args.device, args.resume
    )
    for phase in args.phases:
        if phase == "speed":
            for spec in speed_plan(config):
                campaign.run(spec)
        elif phase == "correctness":
            campaign.correctness()
        else:
            campaign.capacity()


if __name__ == "__main__":
    main()
