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

`./code-polishyw check --all` reports no findings on this branch. Version 3.0.0
passes 227 CPU tests and 227 CUDA tests, including two-GPU NCCL references;
see [current acceptance and remaining checks](docs/plans/modernization.md#progress).
The [development baseline](docs/development-baseline.md) retains the original
adoption evidence.

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
Linux x86_64 machine with two visible NVIDIA GPUs. The lock resolves PyTorch's
CUDA 13 wheels, which need an NVIDIA driver from the 580 series or newer:
`nvidia-smi` must report `CUDA Version: 13.0` or higher. The image's own CUDA
toolkit and Python are not used. Verify the actual runtime first:

```sh
.venv/bin/uv run --frozen --no-sync python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.device_count() >= 2'
TEST_DEVICE=cuda TEST_DTYPES=float16,float32,float64 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/uv run --frozen --no-sync python -m unittest tests.utils.test_distributed
TEST_DEVICE=cuda TEST_DTYPES=float16,float32,float64 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/uv run --frozen --no-sync python -m unittest discover
```

The distributed tests spawn their own ranks; do not launch them with `torchrun`.
The CPU profile runs the two-rank references with Gloo. A requested CUDA run
fails when fewer than two GPUs are available. Record the device/driver versions, test
counts, failures, and skips. GPU provisioning and Linux/CUDA acceptance remain
separate from local Mac verification.

On the two-A40 host with driver 595.91.07, direct GPU-to-GPU transfers hung
even in a bare PyTorch/NCCL probe without library code. For this host, set
`export NCCL_P2P_DISABLE=1` before the CUDA tests and distributed benchmarks.
This disables NCCL's direct peer-to-peer transport while retaining two-device
NCCL execution. Record the setting with results: they do not establish direct
peer-to-peer transport correctness or its performance. Do not apply this
workaround to other hosts unless a separate transport probe shows the same issue.

### Rented GPU hosts

For occasional remote development, keep everything that should persist on
storage that outlives the machine, and rebuild `.venv` from the lock on that
host; a Mac virtual environment is not portable to Linux. On RunPod, create a
network volume once, then create a Pod with two GPUs, that volume, and exposed
TCP port 22 for each session, and terminate the Pod afterwards. The volume mounts
at `/workspace` and survives termination; the rest of the container, including
`/root`, does not. A network volume is tied to its datacenter, and it bills until
deleted. Connect with the Pod's direct TCP address and port.

Create a persistent home on the volume. Tools installed under it, uv's Python
builds and caches, Git configuration, and agent logins then survive Pod
replacement:

```sh
mkdir -p /workspace/home
cat > /workspace/env.sh <<'EOF'
export HOME=/workspace/home
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/pytorch-metric-learning" 2>/dev/null || cd "$HOME"
EOF
. /workspace/env.sh
curl -LsSf https://astral.sh/uv/0.13.4/install.sh | env UV_NO_MODIFY_PATH=1 sh
git clone https://github.com/KevinMusgrave/pytorch-metric-learning
cd pytorch-metric-learning
git switch BRANCH_UNDER_TEST
```

Then follow [Set up a checkout](#set-up-a-checkout) and the CUDA commands above.
Each later session needs only `. /workspace/env.sh` after connecting. Run
`uv sync --frozen --all-extras --no-build --no-install-project` again only when
`uv.lock` changes. Logins saved under the persistent home are stored on the
provider's disk.

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
