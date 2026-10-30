# Development

This guide covers the maintainer workflow: environment, verification, and
releases. [CONTRIBUTING.md](CONTRIBUTING.md) is the shorter guide for pull
requests. Design rationale lives in `docs/design/` and dependency assessments in
`docs/security/`; both are maintainer records, so `mkdocs.yml` excludes them
from the published site.

The development baseline is Python 3.12.13 and uv 0.13.4. The exact
environment is in `.python-version`, `pyproject.toml`, and `uv.lock`; the lock
resolves after `exclude-newer` in `pyproject.toml`, so newer releases are
ignored until that date is moved. There is no test matrix for older PyTorch
versions.

## Set up a checkout

Install the exact uv version, then run from the repository root:

```sh
uv sync --frozen --all-extras --no-build --no-install-project
```

Installation uses published wheels from the frozen lock with source builds
disabled. Tests import the checkout directly. For interactive use, install the
checkout with its already-installed build backend:

```sh
.venv/bin/uv pip install --no-deps --no-build-isolation --editable .
```

Maintainers also run the pinned policy tool, which reviews dependency changes
before installation and runs the merge gate:

```sh
./code-polishyw setup
./code-polishyw dependency-review --base origin/master
uv sync --frozen --all-extras --no-build --no-install-project
./code-polishyw supply-chain
```

## Lint, types, and documentation

```sh
.venv/bin/uv run --frozen --no-sync ruff format --check
.venv/bin/uv run --frozen --no-sync ruff check
.venv/bin/uv run --frozen --no-sync ty check src tests docs/recipes benchmarks scripts
.venv/bin/uv run --frozen --no-sync mkdocs build --strict
```

These are the same pinned tools CI runs. The policy check adds dead-code
detection and module dependency direction from `.code-polishy.json`:

```sh
./code-polishyw check --git-changes
```

Source comments and docstrings are not used. Put design rationale in
`docs/design/` and adapted-code provenance in `THIRD_PARTY_NOTICES.md`.

CI uses the locked policy tool's `test-levels` decision to select its jobs.
Ordinary Markdown-only changes run the documentation merge gate without
installing Python dependencies or running the CPU matrix, package build, or
source checks. Source and policy changes retain those checks. Scheduled and
manually dispatched runs compare against the repository root and retain full
verification.

## CPU tests

Pytest runs every test, including `unittest.TestCase` classes. Third-party
plugin autoloading is disabled so the frozen environment determines behavior.
Each test starts with `torch.manual_seed(0)`. Skipped tests and expected
failures fail the run, with a message explaining why; tests marked
`@pytest.mark.cuda` are deselected, not skipped, on CPU runs.

Tests default to CPU with float16, float32, and float64:

```sh
.venv/bin/uv run --frozen --no-sync --env-file tests/cpu.env python -m pytest
.venv/bin/uv run --frozen --no-sync --env-file tests/cpu.env python -m pytest tests/losses/test_contrastive_loss.py
```

Append `::TestContrastiveLoss::test_contrastive_loss` to select one test, or
use `-k contrastive`. On macOS, the root `conftest.py` sets
`KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` when unset, because the
Faiss and PyTorch wheels bundle separate OpenMP runtimes; see
[retrieval backends](docs/retrieval.md). The policy tool selects tests from
changed files:

```sh
./code-polishyw test --changed --base origin/master
```

## CUDA and distributed verification

The CPU suite cannot establish CUDA correctness. Use a Linux x86_64 host with
two NVIDIA GPUs and a driver from the 580 series or newer (`nvidia-smi` reports
`CUDA Version: 13.0` or higher); the lock resolves PyTorch's CUDA 13 wheels.
Build `.venv` from the lock on that host, then run:

```sh
.venv/bin/uv run --frozen --no-sync python -c 'import torch; assert torch.cuda.device_count() >= 2, torch.cuda.device_count()'
TEST_DEVICE=cuda TEST_DTYPES=float16,float32,float64 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/uv run --frozen --no-sync python -m pytest
```

The distributed tests spawn their own ranks; do not launch them with
`torchrun`. A CUDA run fails when fewer than two GPUs are visible. If a bare
two-rank NCCL all-reduce hangs on a host, set `NCCL_P2P_DISABLE=1` and record
that setting with the results, because it disables direct peer-to-peer
transport.

## Benchmarks

```sh
.venv/bin/uv run --frozen --no-sync python -m benchmarks.run --help
```

See [performance](docs/performance.md) for the measured workloads and how to
reproduce them.

## Releases

1. Update `version` in `pyproject.toml` and run `uv lock`. Write
   `docs/release-notes/VERSION.md` and link it from
   `docs/release-notes/index.md`; the release preflight requires the file, and
   the GitHub release uses it as its notes.
2. Renew dependency assessments in `.code-polishy.json` that would expire
   before the tag. Severity limits make critical assessments last at most seven
   days, so renew those within a week of tagging.
3. Run the full CPU suite and the CUDA suite on a two-GPU host. Record the
   revision, devices, driver, PyTorch build, and test counts in the pull
   request.
4. Merge into `master` and wait for its CI run to succeed.
5. Create an annotated `v<VERSION>` tag on that exact commit and push only the
   tag. Prereleases use PEP 440 spelling, for example `v3.1.0rc1`, and become
   GitHub prereleases.

The [release workflow](.github/workflows/release.yml) checks the tag against
`pyproject.toml`, `master` history, the release notes, and that commit's CI
result. It builds both distributions, verifies the installed wheel in isolated
Python, publishes to PyPI with trusted publishing, creates the GitHub release
with the notes and distributions attached, and deploys the documentation, which
includes the final version 2 documentation under `/v2/`.

If an upload is interrupted, rerun the failed job within the 30-day artifact
retention period; uv skips identical files already on PyPI and rejects
conflicting contents. Documentation and GitHub release jobs can be rerun
independently. Source changes require a new version and tag; never move a
release tag.

### One-time publishing setup

1. Create a GitHub environment named `pypi` in the
   [repository settings](https://github.com/KevinMusgrave/pytorch-metric-learning/settings/environments)
   and allow tags matching `v*.*.*`.
2. Add a trusted publisher in the
   [PyPI publishing settings](https://pypi.org/manage/project/pytorch-metric-learning/settings/publishing/):
   owner `KevinMusgrave`, repository `pytorch-metric-learning`, workflow
   `release.yml`, environment `pypi`.
3. In Settings → Pages, set the source to **GitHub Actions** and allow tags
   matching `v*.*.*` in the `github-pages` environment.
4. Enable private vulnerability reporting under Settings → Security, which
   [SECURITY.md](SECURITY.md) relies on.

No PyPI token or repository secret is required; see the
[PyPI trusted-publisher guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
