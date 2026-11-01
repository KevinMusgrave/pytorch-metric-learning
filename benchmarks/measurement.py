import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

PROXY_CASES = ("arcface", "proxy-anchor")


class MinedTriplet(torch.nn.Module):
    def __init__(self, loss, miner):
        super().__init__()
        self.loss = loss
        self.miner = miner

    def forward(self, embeddings, labels):
        return self.loss(embeddings, labels, self.miner(embeddings, labels))


def workload(request):
    losses = importlib.import_module("pytorch_metric_learning.losses")
    miners = importlib.import_module("pytorch_metric_learning.miners")
    case = request["case"]
    legacy = request["version"] == "2.9.0"
    classes, dimensions = request["proxy_classes"], request["dimensions"]
    factories = {
        "ntxent": lambda: losses.NTXentLoss(temperature=0.07),
        "contrastive": lambda: losses.ContrastiveLoss(pos_margin=0, neg_margin=1.4),
        "triplet": lambda: losses.TripletMarginLoss(margin=0.2),
        "batch-hard-triplet": lambda: MinedTriplet(
            losses.TripletMarginLoss(margin=0.2), miners.BatchHardMiner()
        ),
        "arcface": lambda: losses.ArcFaceLoss(
            classes, dimensions, margin=math.degrees(0.5) if legacy else 0.5, scale=64
        ),
        "supcon": lambda: losses.SupConLoss(temperature=0.1),
        "multi-similarity": lambda: losses.MultiSimilarityLoss(
            alpha=2, beta=50, base=0.5
        ),
        "proxy-anchor": lambda: losses.ProxyAnchorLoss(
            classes, dimensions, margin=0.1, alpha=32
        ),
    }
    dtype, device = getattr(torch, request["dtype"]), request["device"]
    torch.manual_seed(request["seed"])
    module = factories[case]().to(device=device, dtype=dtype)
    if case in PROXY_CASES:
        weight = torch.randn(
            classes,
            dimensions,
            generator=torch.Generator().manual_seed(request["seed"] + 1),
        )
        parameter = proxy_parameter(module, case, legacy)
        with torch.no_grad():
            parameter.copy_(weight.T if case == "arcface" and legacy else weight)
    embeddings = (
        torch.randn(
            request["batch_size"],
            dimensions,
            generator=torch.Generator().manual_seed(request["seed"]),
        )
        .to(device=device, dtype=dtype)
        .requires_grad_()
    )
    labels = torch.arange(request["batch_size"], device=device)
    labels = (
        labels % classes
        if case in PROXY_CASES
        else labels // request["samples_per_class"]
    )
    return module, embeddings, labels


def proxy_parameter(module, case, legacy):
    if not legacy:
        return module.weight
    return module.W if case == "arcface" else module.proxies


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def validate(loss, module, embeddings):
    if not torch.isfinite(loss):
        raise ValueError("nonfinite loss")
    for parameter in (embeddings, *module.parameters()):
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("absent or nonfinite gradient")
    return loss.detach().item()


def timing_events(count, device):
    if device != "cuda":
        return []
    events = [
        tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        for _ in range(count)
    ]
    for group in events:
        for event in group:
            event.record()
    torch.cuda.synchronize()
    return events


def block(module, embeddings, labels, device, count, events):
    module.zero_grad(set_to_none=True)
    embeddings.grad = None
    synchronize(device)
    baseline = 0
    if device == "cuda":
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    forward_ms, backward_ms = 0.0, 0.0
    start = time.perf_counter()
    for index in range(count):
        module.zero_grad(set_to_none=True)
        embeddings.grad = None
        if events:
            events[index][0].record()
        before = time.perf_counter() if not events else 0
        loss = module(embeddings, labels)
        if events:
            events[index][1].record()
        middle = time.perf_counter() if not events else 0
        loss.backward()
        if events:
            events[index][2].record()
        else:
            forward_ms += (middle - before) * 1000
            backward_ms += (time.perf_counter() - middle) * 1000
        last_loss = loss.detach()
        del loss
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if events:
        forward_ms = sum(a.elapsed_time(b) for a, b, _ in events)
        backward_ms = sum(b.elapsed_time(c) for _, b, c in events)
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    return last_loss, {
        "steps": count,
        "elapsed_ms": elapsed_ms,
        "forward_ms": forward_ms / count,
        "backward_ms": backward_ms / count,
        "total_ms": elapsed_ms / count,
        "baseline_allocated_bytes": baseline if device == "cuda" else None,
        "peak_allocated_bytes": peak,
        "peak_incremental_bytes": peak - baseline if peak is not None else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved()
        if device == "cuda"
        else None,
    }


def distribution(values):
    median = statistics.median(values)
    return {
        "mean": statistics.mean(values),
        "median": median,
        "min": min(values),
        "max": max(values),
        "median_absolute_deviation": statistics.median(abs(v - median) for v in values),
    }


def environment(device):
    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "os": platform.platform(),
        "threads": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    if Path("/proc/cpuinfo").exists():
        metadata["cpu"] = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        metadata.update(
            gpu=props.name,
            gpu_total_bytes=props.total_memory,
            compute_capability=list(torch.cuda.get_device_capability()),
            gpu_details=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip(),
        )
    return metadata


def measure(request):
    module, embeddings, labels = workload(request)
    count = request["steps_per_block"]
    events = timing_events(count, request["device"])
    loss, first = block(module, embeddings, labels, request["device"], 1, events[:1])
    value = validate(loss, module, embeddings)
    del loss
    if request["phase"] == "correctness":
        tensors = {"loss": torch.tensor(value), "embeddings": embeddings.grad.cpu()}
        if request["case"] in PROXY_CASES:
            legacy = request["version"] == "2.9.0"
            grad = proxy_parameter(module, request["case"], legacy).grad
            tensors["weight"] = (
                grad.T if legacy and request["case"] == "arcface" else grad
            ).cpu()
        torch.save(tensors, request["gradient_file"])
        return {"loss": value, "first_step": first, "samples": []}
    warmup_steps, start = 0, time.perf_counter()
    while (
        warmup_steps < request["warmup"]
        or time.perf_counter() - start < request["warmup_seconds"]
    ):
        loss, _ = block(module, embeddings, labels, request["device"], 1, events[:1])
        del loss
        warmup_steps += 1
    samples = []
    for _ in range(request["repeats"]):
        loss, sample = block(
            module, embeddings, labels, request["device"], count, events
        )
        samples.append(sample)
        value = validate(loss, module, embeddings)
        del loss
    result = {
        "loss": value,
        "first_step": first,
        "samples": samples,
        "warmup_steps": warmup_steps,
    }
    for key in ("forward_ms", "backward_ms", "total_ms"):
        result[key] = distribution([sample[key] for sample in samples])
    for key in (
        "peak_allocated_bytes",
        "peak_incremental_bytes",
        "peak_reserved_bytes",
    ):
        result[key] = max((s[key] for s in samples if s[key] is not None), default=None)
    return result


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    source = Path(request["source"]).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != request["revision"]:
        raise ValueError("Library checkout does not match the requested revision")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--", "src"], text=True
    )
    if dirty:
        raise ValueError("Library source must be clean")
    sys.path.insert(0, str(source / "src"))
    torch.set_num_threads(request["threads"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if request["device"] == "cuda":
        torch.cuda.set_per_process_memory_fraction(request["memory_fraction"])
    result = {"request": request, "environment": environment(request["device"])}
    try:
        result.update(status="ok", **measure(request))
    except torch.OutOfMemoryError as error:
        result.update(status="oom", error=str(error))
    except ValueError as error:
        if str(error) not in ("nonfinite loss", "absent or nonfinite gradient"):
            raise
        result.update(status="nonfinite", error=str(error))
    except RuntimeError as error:
        if "INT_MAX" not in str(error):
            raise
        result.update(status="index_limit", error=str(error))
    Path(request["result_file"]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
