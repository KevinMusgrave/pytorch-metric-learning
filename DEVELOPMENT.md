# Development

The development baseline is Python 3.12.13, PyTorch 2.14.0, and uv 0.13.4.
Start from `master`. The exact environment is in `.python-version`,
`pyproject.toml`, and `uv.lock`. This checkout does not maintain a test matrix
for older PyTorch versions.

## Set up a checkout

Install the exact uv version, then run from the repository root:

```sh
./code-polishyw setup
./code-polishyw dependency-review --base origin/master
uv sync --frozen --all-extras --no-build --no-install-project
./code-polishyw supply-chain
```

Dependency installation uses published wheels and the frozen lock, with source
builds disabled. The tests import the checkout directly. For interactive use,
install this checkout using its reviewed, already-installed build backend:

```sh
.venv/bin/uv pip install --no-deps --no-build-isolation --editable .
```

`./code-polishyw check --all` reports no findings on this branch. CUDA and
two-GPU acceptance remain outstanding; see [the baseline status](docs/development-baseline.md)
for the original adoption evidence.

## CPU tests

The CPU profile uses float32 and float64 and limits numerical-library
threads. Run a specific test during development:

```sh
.venv/bin/uv run --frozen --no-sync --env-file tests/cpu.env python -m unittest tests.losses.test_contrastive_loss
```

At a completed source boundary, use the configured selection and receipts:

```sh
./code-polishyw test --changed --base origin/master
```

The complete CPU suite is:

```sh
./code-polishyw test --suite cpu-full
```

Agent-facing recipe and sampler integration tests use synthetic tensors and
perform no downloads. The CPU profile also sets `KMP_DUPLICATE_LIB_OK=TRUE`:
on macOS, the Faiss and PyTorch wheels bundle separate OpenMP runtimes, which
coexist only with that setting and single-threaded OpenMP. See
[retrieval backends](docs/retrieval.md#macos-openmp-runtimes).

## CUDA and distributed verification

The CPU suite cannot establish CUDA correctness. For GPU acceptance, use a
Linux machine with two visible NVIDIA GPUs and a driver compatible with the
locked PyTorch CUDA build. Verify the actual runtime first:

```sh
.venv/bin/uv run --frozen --no-sync python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() >= 2'
TEST_DEVICE=cuda TEST_DTYPES=float16,float32,float64 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/uv run --frozen --no-sync python -m unittest tests.utils.test_distributed
TEST_DEVICE=cuda TEST_DTYPES=float16,float32,float64 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/uv run --frozen --no-sync python -m unittest discover
```

The distributed tests spawn their own ranks; do not launch them with `torchrun`.
They skip explicitly under the CPU profile and fail a requested CUDA run when
fewer than two GPUs are available. Record the device/driver versions, test
counts, failures, and skips. GPU provisioning and Linux/CUDA acceptance remain
separate from local Mac verification.

For occasional remote development, retain the checkout and caches on persistent
storage, rebuild `.venv` from the lock on that host, and retain the agent's state
in its configured persistent home. A Mac virtual environment is not portable
to Linux. The Runpod/cloud-devbox lifecycle must verify restart, recreation,
and laptop-disconnected operation before it is relied on for unattended work.

## Policy and packaging

Read [AGENTS.md](AGENTS.md) and the version-matched guidance before edits:

```sh
./code-polishyw docs read agent-workflows
./code-polishyw design-context --git-changes
./code-polishyw format --git-changes
./code-polishyw check --git-changes
```

Source comments and docstrings are disallowed. Design rationale belongs in the
mapped documents under `docs/design/`, and adapted-code provenance in
`THIRD_PARTY_NOTICES.md`. Keep the repository check at zero findings; fix new
findings at their cause rather than suppressing them.

CI owns the final merge gate. Ordinary local task completion uses focused
verification; the merge gate runs at a real merge or release checkpoint.

Build both distributions with the locked backend:

```sh
.venv/bin/uv run --frozen --no-sync python -m build --no-isolation
```

Do not publish a new release until the required policy and platform checks pass.
