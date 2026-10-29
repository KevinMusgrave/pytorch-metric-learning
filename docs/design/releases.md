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

The quick release suite exercises the read-only preflight against temporary
Git repositories and recorded GitHub API responses. Real CI identity, artifact
transfer, and PyPI OIDC exchange are verified only by the hosted workflow.
The one-time account configuration and release steps are in
[Development](../../DEVELOPMENT.md#publishing-to-pypi).

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

Local and CI numerical suites use the same pinned pytest runner. Existing
unittest cases remain supported directly by pytest; its configuration owns
source imports, strict markers, and disabled third-party plugin discovery.
The root `conftest.py` sets macOS OpenMP defaults before importing Torch or the
`tests` package, which itself imports Torch. This makes bare pytest safe with
the pinned Faiss and Torch wheels when OpenMP variables are unset, while
preserving explicit environment settings. A subprocess regression removes
those variables and exercises Faiss indexing, search, and persistence through
ordinary pytest discovery. The library does not set process-wide defaults.
The shared session hook fails skipped and expected-failure cases. Runner tests
execute temporary passing, failing, empty, skipped, and expected-failure suites
to verify the exit codes consumed by CI. CPU and CUDA profiles select tensor
devices and dtypes, while distributed tests continue spawning their own ranks.

The build job also builds the tagged MkDocs site with the pinned development
environment and retains a Pages artifact. After PyPI publication succeeds, a
separate job deploys that artifact through GitHub Pages with `pages: write` and
OIDC permission in the `github-pages` environment. It has no repository-content
write permission and does not rebuild the site. Documentation therefore describes
the released API; a deployment failure can be retried independently. The Pages
source must be GitHub Actions because a `GITHUB_TOKEN` push to a branch does
not trigger another workflow or the legacy Pages build.
