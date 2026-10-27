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

The initial adoption still has existing source findings to resolve. The CUDA
package metadata and PyTorch multiprocessing blockers were fixed by the Code
Polishy upgrade. See [the baseline status](docs/development-baseline.md) for
verification results and remaining work.

## CPU tests

The CPU profile uses float32 and float64, skips external dataset downloads,
and limits numerical-library threads. Run a specific test during development:

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

Agent-facing recipe and sampler integration tests use synthetic tensors. Dataset adapters
have local image/metadata fixtures. Tests that download the original public
datasets remain opt-in through `TEST_DATASETS=true`; they are not required for
the ordinary CPU suite.

## CUDA and distributed verification

The CPU suite cannot establish CUDA correctness. For GPU acceptance, use a
Linux machine with two visible NVIDIA GPUs and a driver compatible with the
locked PyTorch CUDA build. Verify the actual runtime first:

```sh
.venv/bin/uv run --frozen --no-sync python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() >= 2'
TEST_DEVICE=cuda TEST_DTYPES=float16,float32,float64 TEST_DATASETS=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/uv run --frozen --no-sync python -m unittest tests.utils.test_distributed
TEST_DEVICE=cuda TEST_DTYPES=float16,float32,float64 TEST_DATASETS=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/uv run --frozen --no-sync python -m unittest discover
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
mapped documents under `docs/design/`. Existing findings are not exempted by
adoption; policy checks must remain visible until they are resolved.

CI owns the final merge gate. Ordinary local task completion uses focused
verification; the merge gate runs at a real merge or release checkpoint.

Build both distributions with the locked backend:

```sh
.venv/bin/uv run --frozen --no-sync python -m build --no-isolation
```

Do not publish a new release until the required policy and platform checks pass.
