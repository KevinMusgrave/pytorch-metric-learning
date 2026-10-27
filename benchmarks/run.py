import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.distributed import DistributedLossWrapper


@dataclass
class Sample:
    forward_ms: float
    backward_ms: float | None
    peak_allocated_delta_bytes: int | None


@dataclass
class Workload:
    module: nn.Module
    embeddings: torch.Tensor
    labels: torch.Tensor

    def clear_gradients(self) -> None:
        self.module.zero_grad(set_to_none=True)
        self.embeddings.grad = None


class DistributedContrastive(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(dimensions, dimensions, bias=False)
        self.loss = DistributedLossWrapper(losses.ContrastiveLoss(neg_margin=1.4))

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss(self.encoder(inputs), labels)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def step(workload: Workload, device: torch.device) -> tuple[Sample, object]:
    workload.clear_gradients()
    if dist.is_initialized():
        dist.barrier()
    synchronize(device)
    baseline = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    start = time.perf_counter()
    output = workload.module(workload.embeddings, workload.labels)
    synchronize(device)
    forward_ms = (time.perf_counter() - start) * 1000
    backward_ms = None
    if isinstance(output, torch.Tensor):
        start = time.perf_counter()
        output.backward()
        synchronize(device)
        backward_ms = (time.perf_counter() - start) * 1000
    peak = None
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) - baseline
    return Sample(forward_ms, backward_ms, peak), output


def validate(workload: Workload, output: object) -> dict[str, object]:
    if isinstance(output, torch.Tensor):
        if not torch.isfinite(output):
            raise ValueError("Benchmark loss is not finite")
        variables = [workload.embeddings, *workload.module.parameters()]
        for variable in variables:
            if variable.requires_grad and (
                variable.grad is None or not torch.isfinite(variable.grad).all()
            ):
                raise ValueError("Benchmark gradients are absent or not finite")
        return {"loss": output.detach().item()}
    if not isinstance(output, tuple) or len(output) != 4:
        raise ValueError("Pair miner must return four index tensors")
    counts = []
    for indices in output:
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long:
            raise ValueError("Miner indices must be integer tensors")
        if not ((indices >= 0) & (indices < len(workload.embeddings))).all():
            raise ValueError("Miner returned an out-of-range index")
        counts.append(len(indices))
    return {"positive_pairs": counts[0], "negative_pairs": counts[2]}


def make_workload(
    args: argparse.Namespace, case: str, device: torch.device
) -> Workload:
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.manual_seed(args.seed + rank)
    dtype = getattr(torch, args.dtype)
    factories = {
        "contrastive": lambda: losses.ContrastiveLoss(neg_margin=1.4),
        "triplet": lambda: losses.TripletMarginLoss(margin=0.2),
        "miner": lambda: miners.PairMarginMiner(pos_margin=0.2, neg_margin=1.4),
        "proxy": lambda: losses.NormalizedSoftmaxLoss(
            num_classes=args.batch_size // args.samples_per_class,
            embedding_size=args.dimensions,
            temperature=0.1,
        ),
        "distributed": lambda: DistributedContrastive(args.dimensions),
    }
    module = factories[case]().to(device=device, dtype=dtype)
    if case == "distributed":
        module = DistributedDataParallel(
            module, device_ids=[device.index] if device.type == "cuda" else None
        )
    if args.mode == "compile":
        module.compile(fullgraph=True)
    embeddings = torch.randn(
        args.batch_size,
        args.dimensions,
        dtype=dtype,
        device=device,
        requires_grad=True,
        generator=torch.Generator(device=device).manual_seed(args.seed + rank),
    )
    labels = (
        torch.arange(args.batch_size, device=device) + rank * args.batch_size
    ) // args.samples_per_class
    return Workload(module, embeddings, labels)


def describe_samples(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    return {
        "median": median,
        "min": min(values),
        "max": max(values),
        "median_absolute_deviation": statistics.median(abs(x - median) for x in values),
    }


def measure(
    args: argparse.Namespace, case: str, device: torch.device
) -> dict[str, object]:
    workload = make_workload(args, case, device)
    start = time.perf_counter()
    _, output = step(workload, device)
    first_step_ms = (time.perf_counter() - start) * 1000
    validation = validate(workload, output)
    del output
    for _ in range(args.warmup):
        step(workload, device)
    samples = [step(workload, device)[0] for _ in range(args.repeats)]
    backward = [s.backward_ms for s in samples if s.backward_ms is not None]
    return {
        "case": case,
        "rank": dist.get_rank() if dist.is_initialized() else 0,
        "first_step_ms": first_step_ms,
        "validation": validation,
        "forward_ms": describe_samples([s.forward_ms for s in samples]),
        "backward_ms": describe_samples(backward) if backward else None,
        "samples": [asdict(sample) for sample in samples],
    }


def cpu_name() -> str:
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def metadata(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        )
    )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "revision": revision,
        "working_tree_dirty": dirty,
        "os": platform.platform(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dtype": args.dtype,
        "mode": args.mode,
        "fullgraph": args.mode == "compile",
        "threads_per_rank": torch.get_num_threads(),
        "world_size": world_size,
        "backend": dist.get_backend() if dist.is_initialized() else None,
        "local_batch_size": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "dimensions": args.dimensions,
        "samples_per_class": args.samples_per_class,
        "data": "standard normal embeddings; contiguous balanced class labels",
        "seed": args.seed,
        "warmup_steps_after_first": args.warmup,
        "measured_steps": args.repeats,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("all", "contrastive", "triplet", "miner", "proxy", "distributed"),
        default="all",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "float64"), default="float32"
    )
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.samples_per_class < 2
        or args.batch_size < 2 * args.samples_per_class
        or args.batch_size % args.samples_per_class
    ):
        parser.error(
            "Use at least two balanced classes with at least two examples per class"
        )
    if min(args.dimensions, args.repeats, args.threads) < 1 or args.warmup < 0:
        parser.error(
            "Use positive dimensions, repeats, and threads, and nonnegative warmup"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (args.case == "distributed") != (world_size > 1):
        parser.error(
            "Run only --case distributed under torchrun with at least two workers"
        )
    return args


def main() -> None:
    args = arguments()
    torch.set_num_threads(args.threads)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device("cuda", local_rank)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.case == "distributed":
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(seconds=90)
        )
    try:
        cases = (
            ("contrastive", "triplet", "miner", "proxy")
            if args.case == "all"
            else (args.case,)
        )
        records = [measure(args, case, device) for case in cases]
        if dist.is_initialized():
            per_rank: list[object] = [None] * dist.get_world_size()
            dist.all_gather_object(per_rank, records)
            results = per_rank
        else:
            results = [records]
        report = {"metadata": metadata(args, device), "results_by_rank": results}
        if not dist.is_initialized() or dist.get_rank() == 0:
            payload = json.dumps(report, indent=2) + "\n"
            if args.output is None:
                print(payload, end="")
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(payload)
                print(f"Wrote {args.output}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
