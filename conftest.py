import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from tests import TEST_DEVICE, TEST_DTYPES

SKIPPED = pytest.StashKey[bool]()
SKIP_FAILURE = (
    "Skipped or expected-failure tests fail this suite: every collected test must "
    "run. Device-specific tests are deselected elsewhere; optional dependencies "
    "come from "
    "`uv sync --frozen --all-extras`."
)


@pytest.fixture(autouse=True)
def seed_torch():
    torch.manual_seed(0)


def device_unavailable(item: pytest.Item) -> bool:
    if item.get_closest_marker("cuda") and TEST_DEVICE.type != "cuda":
        return True
    return (
        bool(item.get_closest_marker("mps")) and not torch.backends.mps.is_available()
    )


@pytest.hookimpl
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    selected = [item for item in items if not device_unavailable(item)]
    deselected = [item for item in items if device_unavailable(item)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


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


@pytest.hookimpl
def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config):
    if config.stash.get(SKIPPED, False):
        terminalreporter.write_line(SKIP_FAILURE, red=True)
