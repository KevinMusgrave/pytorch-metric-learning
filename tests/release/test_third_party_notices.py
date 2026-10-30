import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "pytorch_metric_learning"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"
LOCAL_PATH = re.compile(r"`([\w./-]+\.py)`")


def table_rows(text):
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and not set(line) <= set("|- "):
            yield cells


def covered_paths(text):
    rows = list(table_rows(text))
    for cells in rows:
        if cells[0].startswith("Upstream source"):
            continue
        for cell in cells[1:]:
            yield from LOCAL_PATH.findall(cell)
    for path in LOCAL_PATH.findall(text):
        if path.startswith("tests/"):
            yield path


class TestThirdPartyNotices(unittest.TestCase):
    def test_covered_files_exist(self):
        paths = sorted(set(covered_paths(NOTICES.read_text())))
        self.assertGreaterEqual(len(paths), 10)
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(
                    (PACKAGE / path).is_file() or (ROOT / path).is_file(), path
                )

    def test_upstream_paths_are_not_checked(self):
        text = (
            "| Upstream source | License | Files |\n"
            "| --- | --- | --- |\n"
            "| [a/b](https://example.invalid), `loss/Upstream.py` | MIT "
            "| `losses/fast_ap_loss.py` |\n"
            "Tests copy `losses.py` and `tests/losses/test_histogram_loss.py`.\n"
        )
        self.assertEqual(
            list(covered_paths(text)),
            ["losses/fast_ap_loss.py", "tests/losses/test_histogram_loss.py"],
        )
