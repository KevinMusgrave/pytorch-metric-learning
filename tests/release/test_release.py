import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "release.py"
NOTES_SCRIPT = ROOT / "scripts" / "release_notes.py"


class TestRelease(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "repository"
        self.root.mkdir()
        self.report = Path(temporary.name) / "ci.json"
        self.git("init", "--quiet", "--initial-branch=master")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "release-test@example.invalid")
        self.git("config", "core.hooksPath", str(self.root / ".git" / "hooks"))
        self.write_version("3.0.0")
        self.commit()
        self.git("update-ref", "refs/remotes/origin/master", "HEAD")
        self.git("tag", "-a", "v3.0.0", "-m", "Release 3.0.0")

    def git(self, *arguments):
        return subprocess.run(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "tag.gpgsign=false",
                *arguments,
            ],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write_version(self, version, notes=True):
        (self.root / "pyproject.toml").write_text(
            f'[project]\nname = "pytorch-metric-learning"\nversion = "{version}"\n'
        )
        if notes:
            notes_dir = self.root / "docs" / "release-notes"
            notes_dir.mkdir(parents=True, exist_ok=True)
            (notes_dir / f"{version}.md").write_text(f"# {version}\n")

    def commit(self):
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "Candidate")
        self.ci_run = {
            "head_sha": self.git("rev-parse", "HEAD"),
            "head_branch": "master",
            "event": "push",
            "path": ".github/workflows/ci.yml",
            "status": "completed",
            "conclusion": "success",
        }

    def preflight(self, *, tag="v3.0.0", runs=None):
        self.report.write_text(
            json.dumps({"workflow_runs": [self.ci_run] if runs is None else runs})
        )
        return subprocess.run(
            [sys.executable, str(SCRIPT), tag, "--ci-runs", str(self.report)],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )

    def assert_rejected(self, **arguments):
        result = self.preflight(**arguments)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr)

    def test_accepts_annotated_version_with_successful_master_ci(self):
        result = self.preflight()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(self.ci_run["head_sha"], result.stdout)
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_accepts_python_prerelease_versions(self):
        self.write_version("3.1.0rc1")
        self.commit()
        self.git("update-ref", "refs/remotes/origin/master", "HEAD")
        self.git("tag", "-a", "v3.1.0rc1", "-m", "Release candidate")
        result = self.preflight(tag="v3.1.0rc1")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_version_mismatch(self):
        self.assert_rejected(tag="v3.0.1")

    def test_requires_release_notes_for_the_version(self):
        (self.root / "docs" / "release-notes" / "3.0.0.md").unlink()
        self.commit()
        self.git("update-ref", "refs/remotes/origin/master", "HEAD")
        self.git("tag", "-f", "-a", "v3.0.0", "-m", "Release 3.0.0")
        self.assert_rejected()

    def test_rejects_lightweight_tags(self):
        self.git("tag", "-d", "v3.0.0")
        self.git("tag", "v3.0.0")
        self.assert_rejected()

    def test_rejects_tag_pointing_to_another_commit(self):
        (self.root / "change.txt").write_text("change\n")
        self.commit()
        self.git("update-ref", "refs/remotes/origin/master", "HEAD")
        self.assert_rejected()

    def test_rejects_tag_outside_master(self):
        (self.root / "change.txt").write_text("change\n")
        self.commit()
        self.git("tag", "-f", "-a", "v3.0.0", "-m", "Candidate off master")
        self.assert_rejected()

    def test_rejects_modified_tracked_files(self):
        self.write_version("3.0.0")
        with (self.root / "pyproject.toml").open("a") as stream:
            stream.write("\n")
        self.assert_rejected()

    def test_rejects_untracked_files(self):
        (self.root / "uncommitted.txt").write_text("change\n")
        self.assert_rejected()

    def test_requires_completed_ci_for_the_exact_candidate(self):
        cases = (
            ("head_sha", "0" * 40),
            ("head_branch", "feature"),
            ("event", "pull_request"),
            ("path", ".github/workflows/release.yml"),
            ("status", "queued"),
            ("status", "in_progress"),
            ("conclusion", "failure"),
            ("conclusion", "cancelled"),
            ("conclusion", "skipped"),
        )
        for key, value in cases:
            with self.subTest(field=key, value=value):
                self.assert_rejected(runs=[{**self.ci_run, key: value}])
        self.assert_rejected(runs=[])

    def test_older_success_cannot_override_latest_failure(self):
        self.assert_rejected(
            runs=[{**self.ci_run, "conclusion": "failure"}, self.ci_run]
        )


class TestReleaseNotes(unittest.TestCase):
    def render(self, root, version):
        output = root / "notes.md"
        subprocess.run(
            [sys.executable, "-I", str(NOTES_SCRIPT), version, str(output)],
            cwd=root,
            check=True,
        )
        return output.read_text()

    def test_relative_links_become_published_site_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs" / "release-notes").mkdir(parents=True)
            (root / "mkdocs.yml").write_text(
                "site_name: x\nsite_url: https://x.io/pml/\n"
            )
            (root / "docs" / "release-notes" / "1.0.0.md").write_text(
                "[a](../migration.md) [b](../recipes/training.md#step) "
                "[c](index.md) [d](#local) [e](https://example.com/page.md)\n"
            )
            self.assertEqual(
                self.render(root, "1.0.0"),
                "[a](https://x.io/pml/migration/) "
                "[b](https://x.io/pml/recipes/training/#step) "
                "[c](https://x.io/pml/release-notes/) [d](#local) "
                "[e](https://example.com/page.md)\n",
            )

    def test_current_notes_have_no_relative_markdown_links(self):
        version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "version"
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "notes.md"
            subprocess.run(
                [sys.executable, "-I", str(NOTES_SCRIPT), version, str(output)],
                cwd=ROOT,
                check=True,
            )
            self.assertNotRegex(output.read_text(), r"\]\((?!https?:|#)[^)]*\.md")


if __name__ == "__main__":
    unittest.main()
