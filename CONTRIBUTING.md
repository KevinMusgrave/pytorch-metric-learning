# Contributing

Bug reports, fixes, documentation improvements, and new losses or miners with
a published reference are welcome. For a larger change, open an issue first so
the design can be agreed before you write code.

## Set up

Install [uv](https://docs.astral.sh/uv/) 0.13.4, then from the repository root:

```sh
uv sync --frozen --all-extras --no-build --no-install-project
```

## Before opening a pull request

```sh
.venv/bin/uv run --frozen --no-sync ruff format
.venv/bin/uv run --frozen --no-sync ruff check
.venv/bin/uv run --frozen --no-sync ty check src tests docs/recipes benchmarks scripts
.venv/bin/uv run --frozen --no-sync --env-file tests/cpu.env python -m pytest
.venv/bin/uv run --frozen --no-sync mkdocs build --strict
```

For source changes, CI runs these commands on Linux and the test suite on macOS
and Windows, plus the maintainers' policy checks described in
[DEVELOPMENT.md](DEVELOPMENT.md). Ordinary Markdown-only changes, such as README
prose, run formatting and documentation checks without the application suites.
Control documents and product-input Markdown receive source verification.
You do not need to install or run the policy tool or edit `.code-polishy.json`.
If a policy step fails on your pull request while the commands above pass, say
so in the pull request and a maintainer will resolve it.

## What a change needs

- Tests that check observable behavior against an independent reference, such
  as a float64 formula or the paper authors' code, including gradients and
  edge cases like empty tuples or missing positives.
- Documentation updated in the same pull request: the component's page under
  `docs/`, and the release notes for user-visible changes.
- A paper citation for a new loss or miner, and the license notice in
  `THIRD_PARTY_NOTICES.md` if you adapted code from another project.
- No comments or docstrings in library source; explain design decisions in the
  pull request or `docs/design/`.

Contributors are credited in [CREDITS.md](CREDITS.md) and keep authorship of
their commits.

Automated coding agents should also follow [AGENTS.md](AGENTS.md).
