## Authority and communication

- Run pinned Code Polishy through `./code-polishyw` (PowerShell:
  `.\code-polishyw.ps1`); use `setup` if the release is absent.
- Before repo changes, run `./code-polishyw docs read agent-workflows`
  and follow its version-matched request-capture and delivery rules.
- During upgrades, outgoing guidance governs until the exact incoming release
  atomically rewrites the lock, activating its guidance.
- `.code-polishy.json` declares modules, dependency direction, capabilities,
  commands, tests, artifacts, and exceptions; it cannot weaken the locked baseline.
- Keep updates outcome-first and under 100 words; detail only for action or
  safety.
- Edit root `## Project principles` only at the caller's explicit request;
  specialization cannot weaken the baseline.

## Implementation

- Preserve unrelated work; avoid unrelated refactors. Prefer the simplest
  end-to-end root-cause fix. Machinery must address demonstrated needs, preserve
  valid results and ordinary recovery, and reduce end-to-end complexity, failure
  modes, and risk.
- Hash only for trust-boundary authentication, immutable identity, or reusable
  evidence. Never hash local state for change detection, mirror an authoritative
  digest, or rehash within one trusted operation.
- Add compatibility, migration, or transitional code only on explicit request.
- Before governed source changes, retrieve `code-polishy design-context` for
  the scope. Reuse it until scope, mappings, or documents change. Follow
  `agent-workflows` for missing rationale and design updates.
- Honor `quality.allowComments`: when false, omit prose comments and docstrings
  from governed handwritten source; when true, preserve useful accurate comments
  and add only what code cannot convey. Put non-local rationale in mapped design
  documents.
- Keep prompt, agent, task, rejection, and editing narration out of final
  artifacts unless that process is their documented subject.
- Remove rejected behavior and its guards, flags, fallbacks, tests, names,
  configuration, and compatibility paths unless final requirements need them.

## Dependencies and tests

- Pin dependencies and package managers exactly; use frozen locks. Generate
  update locks without scripts. Before installation, run
  `code-polishy dependency-review --base <merge-target>`; then install frozen
  with scripts off, complete its reported deferred checks, and test.
- Agents own dependency assessments; no human sign-off. Trace advisory inputs
  to API and impact; presence is not exposure. Retain aged versions if unaffected.
  Admit fixes under 30 days only if waiting is riskier and no aged fix or practical
  mitigation suffices. Document both risks; unknown is not unaffected.
- Keep exceptions exact, visible, owned, justified, and expiring.
- Give each module a quick boundary suite. Test observable behavior with
  temporary state. Reject tautological, change-detector, no-op,
  pass-with-no-tests, and coverage-only tests; checked-in Gherkin must execute.
- Run supplemental suites only when requested or selected by a checked-in
  trigger or release checklist. Declarations, including
  `tests.requiredSupplementalKinds`, never authorize execution. Reuse receipts
  with `test --supplemental --resume`; run all only without trusted evidence or
  after shared infrastructure, toolchain, selection, or unbounded-impact
  changes. Credentialed, destructive, or live-provider probes need named
  external approval.

## Reviews and delivery

- Agent review cannot replace policy checks or required human approval.
- Use caller's checkout for ordinary interactive work; use
  `code-polishy task-session` for unattended work or explicitly requested
  isolation.
- For ordinary Markdown-only work, run `code-polishy format --git-changes`, fix
  its findings, and skip application tests. Verify control and product-input
  Markdown as source.
- During development, run the narrowest useful exact test after a coherent
  runnable change, not after every edit or chat turn. Use
  `code-polishy test --changed` at a completed source boundary only when a final
  gate will not immediately follow. Resolve its base using `agent-workflows`.
- Run `code-polishy merge-gate --base REF` only at a genuine merge or release
  checkpoint through `verification.finalGateOwner`. Ordinary task completion,
  commits, and delivery do not select it. Duplicate only on request; an exact
  pass executes nothing, and only an unchanged failed candidate may resume.
- Commit task-owned progress at milestones, roughly every 1–2 hours of active
  editing on long tasks. Checkpoints may be unfinished or failing; record what
  remains and verification status. Do not wait for gates or API cutovers.
- Before delivery, complete required verification and commit remaining task-owned
  changes unless the caller requests an uncommitted handoff. Public cutovers
  must be coherent at merge or release. Push, publish, and pull-request operations
  require explicit caller authorization.
