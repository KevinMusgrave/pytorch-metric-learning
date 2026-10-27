# Development baseline status

Recorded on 2026-10-26; upgrade verified on 2026-10-27 UTC.
The local Mac environment is usable; Code Polishy
adoption is not yet a passing merge baseline. No Linux/CUDA or live Runpod
acceptance is claimed.

## Environment and policy

- Python 3.12.13, PyTorch 2.14.0, uv 0.13.4.
- Dependency wheels installed with `uv sync --frozen --all-extras --no-build
--no-install-project` after dependency review.
- The local checkout installed editable with the already-installed build
  backend, without dependency resolution or build isolation.
- Code Polishy 0.31.4, release digest
  `9940edf8db1d0e4af006d0f0cb6501323fd9f19795b7f15a9ae4e29179375c89`.
- Source comments/docstrings disabled; full application merge gate; no
  supplemental suites or Gherkin introduced. CI owns the final merge gate.

| Module      | Depends on           | Responsibility                                                      |
| ----------- | -------------------- | ------------------------------------------------------------------- |
| numerics    | none                 | Losses, miners, distances, reducers, regularizers, tensor utilities |
| evaluation  | numerics             | Retrieval, accuracy calculation, testers, logging hooks             |
| training    | numerics, evaluation | Trainers and samplers                                               |
| datasets    | numerics             | Dataset metadata, images, splits, downloads                         |
| distributed | numerics             | Distributed loss/miner wrappers                                     |

No additional capabilities are declared. Python and uv inputs activate their
language and dependency checks; GitHub Actions are pinned and validated.
Five mapped design documents describe the boundaries. The project-local
providers run uv's offline lock check and a distribution build using the
locked setuptools backend, because these are Python packaging operations.

## Verification evidence

The following table records the initial 0.31.3 adoption checks. The upgrade
results below supersede its two tooling failures.

| Command                                                                                                                                                                                                                               | Observed result                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `./code-polishyw doctor --strict`                                                                                                                                                                                                     | Pass                                                                                    |
| `./code-polishyw format --git-changes`                                                                                                                                                                                                | Pass                                                                                    |
| `git diff --check`                                                                                                                                                                                                                    | Pass                                                                                    |
| `.venv/bin/python -m build --no-isolation`                                                                                                                                                                                            | Wheel and source distribution built                                                     |
| `.venv/bin/uv run --frozen --no-sync --env-file tests/cpu.env python -m unittest discover`                                                                                                                                            | 194 tests in 192 seconds; pass, 15 skips                                                |
| Focused rerun of trainer, sampler, inference, local dataset, and wrapper tests after fixture changes                                                                                                                                  | 13 tests passed                                                                         |
| Final trainer checkpoint-recovery and inference rerun                                                                                                                                                                                 | 5 tests passed                                                                          |
| `./code-polishyw check --files tests/trainers/test_metric_loss_only.py tests/samplers/test_tuples_to_weights_sampler.py tests/datasets/test_local_datasets.py tests/utils/test_distributed_fallback.py tests/utils/test_inference.py` | Pass                                                                                    |
| `./code-polishyw test --changed --base origin/master`                                                                                                                                                                                 | Four module suites passed; evaluation test assertions passed but process cleanup failed |
| `./code-polishyw test --suite evaluation-focused`                                                                                                                                                                                     | Reproduced runner failure after unittest reported two passing tests                     |
| `./code-polishyw dependency-review --base origin/master` and `./code-polishyw supply-chain`                                                                                                                                           | Blocked by CUDA package version-normalization finding                                   |
| `./code-polishyw gate`                                                                                                                                                                                                                | Failed on existing source findings before application test/build stages                 |

The full CPU run preceded the final inference fixture and checkpoint-recovery
updates; focused reruns cover those later changes. It does not establish CUDA
coverage. Dataset download tests remain opt-in, and CUDA distributed cases
explicitly skip in the CPU profile. A CUDA distributed request now requires
at least two visible GPUs instead of silently executing an empty loop.

Supplemental hardening: **NOT RUN**, with no selected trigger or declared
suite. Final merge gate: **NOT RUN**; there has been no merge/release checkpoint
or CI publication. No repository debt exceptions were introduced.

## Remaining work

1. Work through existing source debt in bounded changes. The first ordinary
   gate reported over 1,600 findings, including prose comments/docstrings,
   formatting, lint, typing, dead-code reports, complexity, and two import
   cycles. Review findings individually; preserve public exports, required
   upstream attributions, and algorithm behavior while fixing them. A broad
   suppression is not an adoption strategy.
2. Revalidate on Linux with two CUDA devices, then verify persistent remote
   development across restart, machine recreation, and laptop disconnection.
   Provider provisioning and live acceptance have not run in this checkout.

[Dependency assessments](security/dependency-assessments.md) document two exact
fresh security-fix admissions (Pillow through 2026-11-15 and uv through
2026-11-19) and six exact not-affected advisory assessments. The critical certifi
assessment needs review by 2026-11-02; the others expire on 2026-11-25.

## Code Polishy 0.31.4 upgrade verification

The published release was installed through its checksum-pinned publication
index and applied using Code Polishy's managed upgrade plan. The plan reported
no capability changes and no added, removed, or changed diagnostics. Managed
guidance and wrappers were already current.

| Command                                           | Observed result                                                              |
| ------------------------------------------------- | ---------------------------------------------------------------------------- |
| `./code-polishyw version`                         | 0.31.4                                                                       |
| `./code-polishyw doctor --strict`                 | Pass                                                                         |
| `./code-polishyw supply-chain`                    | Pass, including current online security evidence and the local lock provider |
| `./code-polishyw test --suite evaluation-focused` | Two tests passed; process cleanup also passed                                |

The supply-chain check accepts the equivalent CUDA package version spellings.
The evaluation tests retain their existing DataLoader workers and now complete
through Code Polishy's runner. No application dependencies, test commands,
source policy, or assessments were changed to obtain these results.

The [release notes](https://github.com/riteofstring/code-polishy/releases/tag/v0.31.4)
also describe newer Python targets, literal `env` command prefixes, managed
virtualenv interpreter symlinks, and the initial adoption bootstrap. Those four
changes were not separately exercised in this checkout during the upgrade.
The full source gate, full CPU suite, and supplemental suites were not rerun
for this lock-and-documentation change. Existing source findings remain open.

Local verification reports remain under `.code-polishy-reports/`.
