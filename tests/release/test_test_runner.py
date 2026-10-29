import os
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
            "conftest",
            str(candidate),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == expected_exit, result.stdout + result.stderr


def test_bare_pytest_runs_faiss_without_openmp_environment():
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("KMP_DUPLICATE_LIB_OK", None)
    environment.pop("OMP_NUM_THREADS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/utils/test_retrieval_backends.py::TestRetrievalBackends::test_faiss_index_persistence_and_incremental_add",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
