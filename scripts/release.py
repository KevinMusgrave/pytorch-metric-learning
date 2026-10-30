import argparse
import json
import subprocess
import tomllib
from pathlib import Path


def git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_release(root, tag, ci_runs):
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    if tag != f"v{project['version']}":
        raise ValueError("Release tag must match the version in pyproject.toml")
    if not (root / "docs" / "release-notes" / f"{project['version']}.md").is_file():
        raise ValueError("Release notes must exist at docs/release-notes/VERSION.md")
    if git(root, "status", "--porcelain"):
        raise ValueError("Release checkout must be clean")
    reference = f"refs/tags/{tag}"
    if git(root, "cat-file", "-t", reference) != "tag":
        raise ValueError("Release tag must be annotated")
    commit = git(root, "rev-parse", "HEAD")
    if git(root, "for-each-ref", "--format=%(object)", reference) != commit:
        raise ValueError("Release tag must point directly at the checked-out commit")
    try:
        git(root, "merge-base", "--is-ancestor", commit, "origin/master")
    except subprocess.CalledProcessError as error:
        raise ValueError("Release commit must be in origin/master history") from error
    runs = ci_runs.get("workflow_runs", [])
    expected = {
        "head_sha": commit,
        "head_branch": "master",
        "event": "push",
        "path": ".github/workflows/ci.yml",
        "status": "completed",
        "conclusion": "success",
    }
    if not runs or any(runs[0].get(key) != value for key, value in expected.items()):
        raise ValueError(
            "The latest master push CI run for this commit must complete successfully"
        )
    return commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--ci-runs", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        runs = json.loads(arguments.ci_runs.read_text())
        commit = validate_release(Path.cwd(), arguments.tag, runs)
    except subprocess.CalledProcessError as error:
        parser.exit(1, f"Git release check failed: {error.stderr.strip()}\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Release preflight passed for {arguments.tag} at {commit}")


if __name__ == "__main__":
    main()
