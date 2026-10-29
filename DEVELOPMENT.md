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

Version 3.0.0 acceptance includes the full CPU suite, the full CUDA suite with
two-GPU NCCL references, the strict documentation build, and the isolated wheel
check. The [release checklist](#release-checklist) lists what each release
records.

## CPU tests

Pytest is the test runner for local development and CI. Existing
`unittest.TestCase` tests run directly; new tests can use pytest fixtures and
parametrization. Collection adds `src` through the checked-in pytest
configuration. Third-party plugin autoloading is disabled so the frozen
environment determines test behavior. Empty selections, skipped tests, and
expected failures fail verification rather than silently reducing coverage.

Tests default to CPU with float32 and float64. On macOS, the root `conftest.py`
sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` before importing Torch
when these variables are unset, so plain `.venv/bin/python -m pytest` works
with the development wheels. Explicit environment settings take precedence.
The CPU profile also limits numerical-library threads consistently across
platforms. Run a specific test during development:

```sh
.venv/bin/uv run --frozen --no-sync --env-file tests/cpu.env python -m pytest tests/losses/test_contrastive_loss.py
```

Append `::TestContrastiveLoss::test_contrastive_loss` to select one test, or
use `-k contrastive` to filter by name.

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
TEST_DEVICE=cuda TEST_DTYPES=float16,float32,float64 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/uv run --frozen --no-sync python -m pytest tests/utils/test_distributed.py
TEST_DEVICE=cuda TEST_DTYPES=float16,float32,float64 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/uv run --frozen --no-sync python -m pytest
```

The distributed tests spawn their own ranks; do not launch them with `torchrun`.
The CPU profile runs the two-rank references with Gloo. A requested CUDA run
fails when fewer than two GPUs are available. Record the device/driver versions, test
counts, failures, and skips. GPU provisioning and Linux/CUDA acceptance remain
separate from local Mac verification.

On one two-A40 host with driver 595.91.07, direct GPU-to-GPU transfers hung
even in a bare PyTorch/NCCL probe without library code. For this host, set
`export NCCL_P2P_DISABLE=1` before the CUDA tests and distributed benchmarks.
This disables NCCL's direct peer-to-peer transport while retaining two-device
NCCL execution. Record the setting with results: they do not establish direct
peer-to-peer transport correctness or its performance. Do not apply this
workaround to other hosts unless a separate transport probe shows the same issue.
A later two-A40 host with the same driver passed both a bare two-rank NCCL
probe and the complete CUDA suite with `NCCL_P2P_DISABLE` unset. The need for
this setting depends on the host, not just the GPU model and driver version.

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
`THIRD_PARTY_NOTICES.md`. The check includes formatting, linting, type checking
of the annotated public surface, dead-code detection, and module dependency
direction from `.code-polishy.json`. Keep it at zero findings; fix new findings
at their cause rather than suppressing them. Formatting and lint rules are also
available to plain `ruff` through `[tool.ruff]` in `pyproject.toml` for
contributors without the policy tool.

CI owns the final merge gate. Ordinary local task completion uses focused
verification; the merge gate runs at a real merge or release checkpoint.

Build both distributions with the locked backend:

```sh
.venv/bin/uv run --frozen --no-sync python -m build --no-isolation
```

Do not publish a new release until the required policy and platform checks pass.

## Release checklist

Before tagging a release:

1. Run the full CPU suite and the CUDA suite on a two-GPU host, and record the
   revision, device, driver, PyTorch build, test counts, and any skips in the
   release notes or pull request.
2. Build the documentation with `mkdocs build --strict` and check a page with
   equations on GitHub and on the built site.
3. Confirm the dated dependency assessments in `.code-polishy.json` are valid
   through the release date; CI runs the dependency review weekly and fails
   when one expires.
4. Merge into `master`, wait for its CI run, then create and push the annotated
   tag as described below.

## Publishing to PyPI

[The release workflow](.github/workflows/release.yml) publishes a wheel and a
source distribution when an annotated version tag such as `v3.0.0` is pushed.
The tag must match `pyproject.toml`, point at a commit in `master` history, and
have a successful `master` push run of the CI workflow for that exact commit.
The release workflow checks that result and refreshes dependency policy; it
does not repeat the ordinary merge gate.

Configure trusted publishing once:

1. Create a GitHub environment named `pypi` in
   [repository settings](https://github.com/KevinMusgrave/pytorch-metric-learning/settings/environments).
   Allow release tags matching `v*.*.*`. Leave required reviewers unset for
   automatic publication after the workflow checks pass.
2. Add a GitHub trusted publisher in the project's
   [PyPI publishing settings](https://pypi.org/manage/project/pytorch-metric-learning/settings/publishing/):

   | Field             | Value                     |
   | ----------------- | ------------------------- |
   | Owner             | `KevinMusgrave`           |
   | Repository        | `pytorch-metric-learning` |
   | Workflow filename | `release.yml`             |
   | Environment       | `pypi`                    |

No PyPI API token or repository secret is required. The publisher uses a
short-lived OIDC identity, following the
[PyPI trusted-publisher setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [uv publishing workflow](https://docs.astral.sh/uv/guides/integration/github/#publishing-to-pypi).

In repository Settings → Pages, set the source to **GitHub Actions**. Allow
release tags matching `v*.*.*` in the `github-pages` environment. After PyPI
publication, the release workflow deploys the site artifact built from the
same tag using pinned MkDocs Material. No local `mkdocs gh-deploy` step or
repository-content write token is required.

Published runtime and optional dependency requirements are compatibility lower
bounds. Their exact development versions live in the `dev` dependency group and
`uv.lock`; build tools and package managers remain exactly pinned. CI verifies
the wheel in a separate core-only environment at the declared minimum versions.
If a development upgrade raises locked versions above the supported floor,
retain a separate minimum-version test environment before making that change.

For each release, update the version in `pyproject.toml` and
`src/pytorch_metric_learning/__init__.py`, regenerate `uv.lock` with the pinned
`uv lock`, and complete the release notes and platform checks. Merge the
candidate into `master` and wait for its CI run to succeed. Create an annotated
`v<VERSION>` tag on that exact commit and push only that tag. Python prerelease
versions use PEP 440 spelling, for example `v3.1.0rc1`.

The build job installs frozen dependencies, builds with the locked backend,
installs the wheel, and checks its version and a loss backward pass in isolated
Python. The publish job downloads those verified distributions and runs
`uv publish --trusted-publishing always` in the `pypi` environment.

If an upload is interrupted, rerun the failed publish job within the artifact's
30-day retention period. It reuses the same files; uv skips identical files
already on PyPI and rejects conflicting contents. Keep dependency assessments
current at release time. Source changes require a new version and tag rather
than moving an existing release tag.

If only documentation publication fails, rerun the failed docs job within the
artifact retention period. It deploys the same site artifact and leaves the
already published PyPI files unchanged.
