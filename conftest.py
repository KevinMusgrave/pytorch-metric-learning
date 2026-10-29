import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest
import torch

from tests import TEST_DEVICE, TEST_DTYPES

SKIPPED = pytest.StashKey[bool]()


@pytest.hookimpl
def pytest_report_header():
    return f"PyTorch {torch.__version__}; device={TEST_DEVICE}; dtypes={TEST_DTYPES}"


@pytest.hookimpl(wrapper=True)
def pytest_report_teststatus(report, config: pytest.Config):
    if report.skipped:
        config.stash[SKIPPED] = True
    return (yield)


@pytest.hookimpl
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if exitstatus == pytest.ExitCode.OK and session.config.stash.get(SKIPPED, False):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
