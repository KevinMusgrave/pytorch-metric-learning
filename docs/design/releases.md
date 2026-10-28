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
