import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release.py"


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

    def write_version(self, version):
        (self.root / "pyproject.toml").write_text(
            f'[project]\nname = "pytorch-metric-learning"\nversion = "{version}"\n'
        )

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


if __name__ == "__main__":
    unittest.main()
