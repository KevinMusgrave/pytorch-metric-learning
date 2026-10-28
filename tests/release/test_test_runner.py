import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("source", "expected_exit"),
    [
        ("def test_directory(tmp_path):\n    assert tmp_path.is_dir()\n", 0),
        ("def test_directory(tmp_path):\n    assert tmp_path.is_file()\n", 1),
        ("", 5),
        (
            (
                "import pytest\n"
                "def test_skip():\n    pytest.skip('unsupported environment')\n"
            ),
            1,
        ),
        (
            (
                "import pytest\n"
                "def test_expected_failure():\n    pytest.xfail('known failure')\n"
            ),
            1,
        ),
        (
            (
                "import unittest\n"
                "class TestSubtest(unittest.TestCase):\n"
                "    def test_partial(self):\n"
                "        with self.subTest(case='unsupported'):\n"
                "            self.skipTest('unsupported environment')\n"
            ),
            1,
        ),
    ],
    ids=["pass", "failure", "empty", "skip", "xfail", "subtest-skip"],
)
def test_runner_exit_status(tmp_path, source, expected_exit):
    candidate = tmp_path / "test_candidate.py"
    candidate.write_text(source)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(root / "pyproject.toml"),
            "-p",
            "tests.conftest",
            str(candidate),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == expected_exit, result.stdout + result.stderr
