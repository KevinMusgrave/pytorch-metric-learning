# Release publication

An annotated `v<VERSION>` tag selects one committed package version for PyPI.
The tag must match `pyproject.toml`, point directly at the checked-out commit,
and belong to `master` history. A clean checkout and the latest successful
`master` push run of `.github/workflows/ci.yml` for that exact commit establish
the release candidate. Missing, pending, failed, skipped, or unrelated CI
results block publication. The release workflow consumes the CI result instead
of duplicating the CI-owned merge gate.

The build job reviews dependencies, installs the frozen wheel-only environment,
refreshes online supply-chain checks, and builds a source distribution and a
wheel with the locked backend. An isolated Python process imports the installed
wheel and checks its version and a differentiable loss. This catches packaging
errors without importing the source checkout. Numerical and distributed suites
remain owned by ordinary CI and the documented platform verification.

Both distributions are retained in one workflow artifact. Only the subsequent
publish job receives `id-token: write` and uses the `pypi` GitHub environment.
It downloads those files without checking out or rebuilding source, then uses
the pinned uv release with trusted publishing required. PyPI binds that
short-lived identity to the repository, workflow filename, and environment;
the repository contains no PyPI API token.

Retry a failed publish job using the same retained artifact. PyPI publication
through uv skips already uploaded identical files and rejects conflicting
content. Release concurrency keeps two attempts for one tag from running
together. Published versions and tags remain immutable; source corrections
receive a new version. These rules preserve recovery from a partial upload
without mixing artifacts from different builds.

Real CI identity, artifact transfer, and PyPI OIDC exchange are verified only
by the hosted workflow. The release steps and one-time account configuration
are in [Development](../../DEVELOPMENT.md#releases).

Published Python requirements declare compatibility bounds, while development,
build, and release environments remain exactly pinned and frozen. Current
lower bounds match the tested versions. CI installs the built wheel in a
separate environment containing only its core runtime dependencies, verifies
that those versions equal the published lower bounds, and runs a differentiable
loss in isolated Python. Future development dependency upgrades must retain
minimum-version coverage instead of silently testing only the newer stack.
Code Polishy's `supplyChain.pythonLibraries` names `pyproject.toml` explicitly;
the policy validates every published requirement against its exact development
pin and still audits the entire frozen dependency graph.

CI runs on pull requests, `master` pushes, and a weekly schedule, so expiring
dependency assessments fail before a release attempt discovers them. The policy
gate runs once on the locked Python 3.12 development environment. Before it,
plain `ruff`, `ty`, and a strict MkDocs build run as separate steps with the
same pinned tools, so contributors without the policy tool get readable
failures, and documentation errors surface in pull requests rather than at
release time. Further jobs run the CPU suite under Python 3.13 and 3.14 and on
macOS and Windows, so the published Python range and the supported CPU
platforms are exercised without a second policy installation. The shared test
session fails skipped and expected-failure cases and prints why; CUDA-only
tests are deselected on CPU runs instead of returning early.

The build job also builds the tagged MkDocs site with the pinned development
environment, adds the final version 2 documentation (tag `v2.9.0`) under `/v2/`
so existing users keep a stable destination after the site moves to version 3,
and retains a Pages artifact. After PyPI publication succeeds, a
separate job deploys that artifact through GitHub Pages with `pages: write` and
OIDC permission in the `github-pages` environment. It has no repository-content
write permission and does not rebuild the site. Documentation therefore describes
the released API; a deployment failure can be retried independently. The Pages
source must be GitHub Actions because a `GITHUB_TOKEN` push to a branch does
not trigger another workflow or the legacy Pages build.

The preflight requires `docs/release-notes/VERSION.md`. After publication, a
job with `contents: write` creates the GitHub release for the tag with those
notes and the verified distributions attached, marking PEP 440 prereleases as
prereleases. The version is declared once, in `pyproject.toml`; the package
does not repeat it in `__version__`, because a dynamic version would require
running the build backend while locking.
