# PyTorch Metric Learning modernization plan

Status: in progress. Phase 1 principles and component decisions are recorded;
implementation and verification of phases 2–5 remain outstanding.

## Purpose

Provide trustworthy, efficient, composable metric-learning implementations
that fit naturally into ordinary PyTorch code.

The library should save users and their coding agents the work of reproducing
papers, validating gradients, optimizing numerical code, and assembling useful
metric-learning components. Its value depends on correctness, performance,
customizability, and clear packaging.

Treat this as a focused major-version modernization. Preserve useful
mathematical ideas and behavior; old class hierarchies, development practices,
dependencies, and PyTorch compatibility are not requirements. Deliver bounded
changes with evidence rather than attempting an unverified rewrite of the
whole library.

## Project principles to add to AGENTS.md

Add a concise `## Project principles` section to the root
[AGENTS.md](../../AGENTS.md) during phase 1. Keep the managed Code Polishy
guidance intact.

1. **Correctness includes gradients.** Define mathematical behavior and test
   values, gradients, numerical stability, edge cases, and distributed
   semantics. A successful training run alone is insufficient evidence.
2. **Use PyTorch directly.** Prefer PyTorch tensor operations, autograd,
   modules, and distributed primitives. Additional abstractions, dependencies,
   or custom kernels must solve a demonstrated problem. Fit naturally with
   device/dtype handling, mixed precision, and compilation where supported.
3. **Measure performance.** Evaluate forward and backward time and peak memory
   on representative workloads. Avoid unnecessary synchronization, host/device
   transfers, and large intermediate tensors. Performance changes must retain
   the documented mathematical behavior within justified numerical tolerances.
4. **Make customization straightforward.** Provide explicit composition points,
   sensible defaults, and clear supported combinations. Changing a distance,
   reduction, or mining strategy should not require understanding an internal
   inheritance framework. Add configuration for meaningful choices.
5. **Respect papers while improving defaults.** Keep the paper-faithful
   configuration reproducible. Make justified algorithmic improvements
   configurable and enable them by default. Document deviations, the rationale
   for the default, and the settings that recover the paper's behavior. Test
   both configurations. Claims of improved training quality require evidence;
   an assumption that a variant is better is not enough.
6. **Keep the library focused.** Retain functionality that provides distinctive
   metric-learning value. Every public feature, abstraction, and dependency
   must justify its maintenance cost. Let applications own ordinary training
   orchestration and logging.
7. **Make behavior easy to discover.** Use clear signatures, types, tensor
   contracts, and executable examples. Document normalization, reduction,
   gradients, shapes, devices, and dtypes. Prefer explicit composition over
   hidden conventions, global state, or class-name inspection.
8. **Cite and give credit.** Cite papers, credit upstream repositories and
   contributors, and preserve required copyright and license notices in
   distributed artifacts. Keep the provenance of adapted algorithms and code
   discoverable through refactors.
9. **Design for agent use.** Make APIs and documentation easy to discover,
   retrieve in small relevant sections, and verify through executable examples.
   State defaults, tensor contracts, supported combinations, and failure behavior
   explicitly. Keep one canonical documentation source shared by agent and human
   interfaces.

Source comments and docstrings remain disallowed by the selected repository
policy. Preserve mathematical explanations in documentation and retain required
upstream attribution and license notices in appropriate distributable files.

## Intended scope

| Area                                                  | Direction                                              | What must be preserved or established                                               |
| ----------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| Losses, miners, distances, reductions, regularization | Retain the useful concepts and simplify implementation | Correct mathematical contracts and practical composition                            |
| Metric-learning samplers                              | Retain where they provide domain-specific value        | Label/batch semantics and reproducible behavior when seeded                         |
| Retrieval metrics                                     | Retain the mathematical functionality                  | Query/reference semantics, self-match exclusion, ties, and empty cases              |
| Distributed support                                   | Audit and improve                                      | Explicit supported combinations and verified gradient semantics                     |
| General training framework                            | Remove from the library                                | Extract valuable algorithm-specific recipes before deleting orchestration           |
| record-keeper integration and logging orchestration   | Remove                                                 | Keep useful optional statistics without requiring a logging framework               |
| Testers, inference wrappers, dataset downloaders      | Remove orchestration; retain retrieval primitives      | Preserve hierarchical/two-stream evaluation recipes and optional retrieval backends |

Retiring the record-keeper integration does not itself archive the separate
record-keeper repository. Do not introduce compatibility shims or maintain old
PyTorch support without a separate explicit requirement. Explain breaking
changes in the next major release's documentation.

## Starting evidence

The [development baseline](../development-baseline.md) records the environment,
CPU results, dependency assessments, and remaining source-policy findings.
Code Polishy 0.31.4 resolves the two tooling blockers found during adoption.
Linux/CUDA and live two-GPU acceptance are still outstanding.

Two local probes identified concrete distributed-wrapper issues:

- `DistributedMinerWrapper` calls `get_world_size()` without checking whether
  a process group exists. Its behavior is inconsistent with the loss wrapper's
  non-distributed fallback.
- `DistributedLossWrapper(CrossBatchMemory(...))` takes the ordinary loss
  fallback when no process group exists and passes too many positional
  arguments to `CrossBatchMemory.forward`.

The [distributed documentation](../distributed.md) also states that
`efficient=True` changes gradients. This is an existing documented algorithmic
choice, not a newly demonstrated numerical bug. It needs an explicit contract
and clearer treatment than a switch that appears to affect only performance.

The current loss base class discovers regularizers by walking the inheritance
hierarchy and matching class names. This is a concrete candidate for
simplification; the usefulness of distance/miner/reducer composition should
survive that change.

## Phase 1: establish principles and component decisions

- Add the project principles to `AGENTS.md` without weakening the locked policy.
- Inventory public components, their dependencies, recipes, and tests. Decide
  which testers, inference helpers, and dataset adapters justify retention.
- Identify metric-specific algorithms embedded in trainers that deserve a
  small executable recipe or an independent component.
- Classify relevant open issues and pull requests against the retained scope:
  correctness, performance, customization, or functionality being retired.
  Use substantive reports to prioritize subsequent changes.
- Record supported behavior and deliberate breaking changes. Keep the public
  API small enough to explain and test.

Completion: the project principles are checked in and every major subsystem has
a concrete retain, simplify, move, or remove decision with a short rationale.

Phase 1 decisions and the issue/PR priorities are recorded in
[component decisions](component-decisions.md). These decisions describe the
target surface; they do not imply that removals or fixes have shipped.

## Phase 2: establish correctness and performance references

- Turn the two reproduced wrapper issues into regression tests and fix their
  dispatch/fallback behavior. Cover queue updates and `enqueue_mask` handling
  for cross-batch memory where applicable.
- Define supported distributed combinations and the meaning of each mode.
  Compare loss values, embedding gradients, and trainable loss-parameter
  gradients where supported against an independent single-process reference.
- Exercise no process group, an initialized one-rank group, and two ranks.
  Cover uneven and empty rank batches, explicit reference embeddings, supplied
  mining indices, reduction/scaling, and relevant memory-queue behavior.
  Unsupported combinations must fail clearly instead of silently changing
  semantics or hanging.
- Obtain two-GPU NCCL evidence through the remote development environment.
  CPU/Gloo checks are useful additional evidence but cannot establish CUDA
  correctness. CPU work can proceed while that environment is prepared.
- Strengthen numerical references for the retained core using small independent
  formulas, analytical examples, and gradient checks where mathematically
  appropriate. Include empty valid sets, repeated labels, normalization,
  zero-distance cases, extreme inputs, and relevant dtypes.
- Establish small repeatable benchmarks for representative pair/triplet losses,
  mining, a learnable proxy loss, and distributed execution. Record hardware,
  software, batch shape, embedding dimension, label distribution, dtype,
  warm-up, synchronization, forward/backward time, and peak memory. Measure
  compilation separately where applicable. Avoid noisy universal thresholds
  before a stable reference exists.

Completion: the known wrapper regressions pass, distributed semantics are
explicit, and retained core behavior has useful correctness references and
performance measurements. Record any unavailable GPU evidence as outstanding.

## Phase 3: remove unwanted subsystems

- Remove the general trainer framework and record-keeper integration after
  extracting any retained metric-specific recipes.
- Remove dependencies used only by retired functionality. For example, evaluate
  TensorBoard and other logging-only dependencies once their callers disappear.
- Apply the phase-1 decisions to testers, inference helpers, and dataset
  downloaders. Keep dependency-heavy retrieval backends optional where useful.
- Remove associated tests, obsolete public exports, examples, documentation,
  configuration, and compatibility paths together. Preserve independent
  numerical tests that remain valuable.
- Update Code Polishy ownership, design documents, dependency declarations,
  frozen locks, and focused suites to describe the surviving architecture.

Completion: retained modules have no dependency on removed subsystems; the
package builds and imports with its declared dependencies; affected behavioral
tests pass; documentation describes the actual supported surface.

## Phase 4: simplify the core with measured evidence

- Start with one pair-based loss and one learnable proxy loss to test the design
  against both stateless numerical work and trainable parameters.
- Prefer small `torch.nn.Module` implementations and explicit composition.
  Replace class-name inspection, incidental inheritance, and initialization
  ordering tricks when a simpler design preserves the useful contracts.
- Make intermediate loss data and reduction semantics explicit. Introduce a
  functional API or a new shared abstraction only when concrete use cases show
  that it reduces complexity; do not create parallel APIs automatically.
- Keep optional statistics off the default hot path and avoid forcing device
  synchronization merely to expose diagnostic values.
- Compare the prototype with phase-2 references and benchmarks before extending
  it to other losses. Investigate material regressions and record any justified
  tradeoff. Check paper-faithful and improved-default configurations separately.
- Extend the proven design in bounded groups, then validate stateful memory and
  distributed composition. Remove superseded implementations and scaffolding.

Completion: representative implementations are simpler to use and maintain,
their values and gradients match the declared contracts, and performance and
memory behavior are measured. Extend only the design that meets those criteria.

## Phase 5: refresh documentation and finish policy adoption

- Make the README a concise introduction, installation entry point, and tested
  minimal example. Use a canonical home for detailed guidance to avoid drift
  between the README and documentation index. Keep it understandable to humans:
  plain-language purpose, installation, a minimal working example, and links by
  task. Human and agent readers use the same canonical documentation; the README
  is not a machine-only manifest or a contributor instruction sheet.
- Treat the repository URL as the primary discovery entry point. An agent given
  only `https://github.com/KevinMusgrave/pytorch-metric-learning` must be able to
  follow the README to a task/component index, select the right public API,
  retrieve its defaults and tensor contracts, install it, and verify a minimal
  integration. Keep library-use guidance distinct from contributor workflows;
  using the package must not require reading the internal class hierarchy or
  repository maintenance instructions. Verify this path against the final tree.
- Document installation from PyPI using `uv` or `pip`, and maintain wheel/sdist
  packaging. Community-maintained downstream distributions are independent of
  the repository's supported installation guidance.
- Document public contracts, customization, paper references, improved defaults,
  paper-faithful settings, and measured performance guidance.
- Organize canonical Markdown into small, searchable topics with stable headings
  and a component/task index. Put small, complete agent-facing executable recipes
  beside the relevant contracts, with explicit inputs, expected results, and
  verification commands. Do not maintain a separate human-facing example gallery
  or tutorial collection. Make installed/public signatures and types agree with those docs;
  an agent should not need to inspect the internal hierarchy to use an API.
- Keep the documentation website as a generated rendering of the same Markdown,
  with readable static content and links to sources. Establish content and
  navigation first. Add a documentation CLI only if a demonstrated lookup or
  offline installed-package need justifies it; it must retrieve the same content
  rather than create a second set of documentation.
- Replace every equation image with editable LaTeX in the canonical Markdown.
  Use MathJax on the website and GitHub's native math rendering where equations
  appear in the README. Verify transcriptions against the paper and implemented
  conventions, including normalization and reduction. Pin and review the math
  renderer like other dependencies, inspect rendered equations in a browser,
  and remove obsolete raster assets once their references are gone.
- Completely redesign the main README architecture diagram for both correctness
  and visual quality after the component boundaries settle. Use a clear reading
  order, restrained colors, consistent typography, generous spacing, and an
  editable vector/diagram source. Show the retained library's relationship to
  ordinary PyTorch code. Check the README and website renderings at narrow and
  wide widths; update or remove the separate loss-flow diagram if it conflicts
  with the final design.
- Keep only agent-facing executable recipes using ordinary PyTorch training loops
  where they demonstrate retained contracts or algorithm-specific composition.
  Remove all Jupyter notebooks and the standalone examples gallery. Remove
  obsolete tutorials, broken downloads, redundant pages, and unused
  assets after checking their references. Preserve useful mathematical material
  and required attribution.
- Fix findings as retained modules change, then complete the remaining Code
  Polishy cleanup after deletion and restructuring. Review dead-code and typing
  reports against intended public behavior; do not delete useful exports simply
  because a local caller is absent.
- Address tool false positives at their cause and keep genuine findings visible.
  Do not hide debt with broad exclusions or weaken the selected baseline.
- Build the documentation and distributions, execute maintained agent-facing recipes, and
  complete the required CPU, CUDA, and two-GPU verification before release.
  At the actual merge/release checkpoint, run the final gate through its
  configured CI owner.

Completion: documentation matches the final API, maintained agent-facing recipes execute,
no Jupyter notebooks or separate examples gallery remain,
the repository-URL discovery path reaches installation, task selection, exact
contracts, and runnable verification without internal implementation knowledge,
equations render from LaTeX without equation images, the redesigned architecture
diagram is accurate and visually verified, required verification passes,
remaining policy findings are resolved, and the major-release notes explain
removals and changed defaults.

## Delivery sequence

Start with phase 1, then the wrapper regressions and numerical/performance
references in phase 2. Once component decisions are settled, deletion work can
proceed while GPU acceptance is arranged. Establish the relevant reference
behavior before replacing each retained implementation.

Use bounded commits with verification appropriate to the change. Dependency
changes follow the pinned review and frozen-install workflow. Ordinary prose
changes need formatting and documentation checks; source changes need their
affected tests. Supplemental suites run only when explicitly selected under
repository policy. Existing findings do not justify polishing components
scheduled for removal.

Defer existing contributor PRs, including #763 and #764, until this modernization
branch is merged into `master`. Then update and verify each original PR against
the modernized baseline and merge it separately, preserving its authorship and
review history. Track the reported defects during modernization; do not absorb
those patches into this branch. The detailed priorities are in
[component decisions](component-decisions.md).

Track implementation progress against this plan and keep current architectural
rationale in `docs/design/`. This plan is a work sequence, not a replacement
for those design documents or evidence that the listed work is complete.

## Progress

- Phase 1: principles added, including attribution; component inventory,
  dependency direction, recipe requirements, and open-work priorities recorded.
- Phase 2: in progress. Seven CPU fallback tests pass, including a real one-rank
  Gloo group, reference gradients, mined pairs, and memory queue wraparound with
  enqueue masks. Two-rank Gloo references now verify full-batch values and
  encoder/proxy DDP gradients, uneven and empty ranks, explicit references,
  supplied mined indices, memory queue ordering, local-anchor semantics, and
  collective argument failures in float32/float64. The reported world-size
  over-scaling and rank-dependent queue ordering are fixed. Broader numerical
  references, benchmarks, and CUDA/NCCL evidence remain outstanding. Contributor
  PRs are deferred until after the modernization merge as described above.
- Cross-batch memory: empty enqueues and batches preserve queue state; references
  remain stable across multiple forwards before backward; masked enqueues cannot
  exceed capacity. Queue metadata now persists in module checkpoints so restored
  training matches uninterrupted values and gradients. Six focused state tests
  cover these contracts. The full CPU suite passed 200 tests with 11 skips before
  the checkpoint-state addition; affected memory tests cover that later change.
- Verification: the distributed milestone passed all five affected focused
  suites (26 tests, one statistics-related skip). Its checkpoint gate still
  reported 1,645 findings in the wider repository; the changed distributed files
  passed their source checks. No merge/release acceptance is claimed.
- Dynamic soft margin: reproduced backward failures in the paired path and on
  successive training steps, plus an out-of-bounds histogram update at maximum
  distance. Detached, checkpointed histogram state, endpoint-safe interpolation,
  normalized batch momentum, and non-mutating matched-pair mining now have
  independent value and gradient references. Unequal streams use only actual
  matched pairs; supplied triplets retain reference gradients without labels.
  Eight reference tests pass on CPU in float16, float32, and float64, and all
  five affected focused suites pass (40 tests, one statistics-related skip).
- Performance tooling: a separate benchmark module now measures representative
  pair, triplet, miner, proxy, and distributed workloads, with raw samples and
  explicit hardware, data, dtype, timing, compilation, and memory semantics.
  Core smoke workloads and a two-process Gloo run pass. All six affected focused
  suites pass. Clean-revision CPU baselines cover batches 128 and 512, two-rank
  Gloo, and full-graph proxy compilation. The larger triplet forward takes
  341.808 ms versus 6.076 ms at batch 128, identifying cubic candidate enumeration
  as an optimization target. Compilation does not improve this small CPU proxy
  workload. CUDA/NCCL evidence remains outstanding. Commands and raw results are
  in [performance measurements](../performance.md).
- Triplet enumeration now expands actual positive pairs by their negative
  counts, preserving ordered indices, self/reference semantics, loss values,
  and gradients without an unconditional cubic candidate mask. Independent
  references cover exhaustive small label sets, empty and unequal references,
  label dtypes, noncontiguous inputs, and embedding/reference gradients. The
  full CPU suite passes 208 tests with 11 skips. Clean-revision measurements
  confirm batch-512 forward time falls from 341.808 ms to 15.358 ms and batch-128
  from 6.076 ms to 1.032 ms, with backward roughly unchanged. Raw measurements
  and limitations are recorded in the performance document.
- Phase 3: training orchestration, logging hooks, model-aware weighted sampling,
  and their private helpers are removed after extracting the numerical
  composition into agent-facing recipes. Six recipe tests verify optimization,
  cascaded candidate filtering, analytical adversarial values/gradients,
  alternating parameter ownership, subset weights, and memory warmup. Testers,
  model inference orchestration, and dataset adapters still await removal.
- Phase 5: all Jupyter notebooks and the separate example gallery are removed.
  The README now provides a human-readable introduction and minimal training
  step, leading to a shared task/component index and agent-facing executable
  recipes beside their contracts. Equation conversion, API documentation,
  diagram redesign, and complete policy adoption remain outstanding.
