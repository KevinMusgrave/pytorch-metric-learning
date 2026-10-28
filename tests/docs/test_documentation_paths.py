import math
import re
import runpy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = [
    ROOT / "README.md",
    *sorted(
        path
        for path in (ROOT / "docs").rglob("*.md")
        if path.relative_to(ROOT / "docs").parts[0]
        not in ("plans", "design", "security")
    ),
]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)|!\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
FENCE = re.compile(r"```.*?```", re.DOTALL)


def anchor(heading):
    text = re.sub(r"[`*_]", "", heading).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text)


def anchors(path):
    return {anchor(h) for h in HEADING.findall(FENCE.sub("", path.read_text()))}


def relative_links(path):
    text = FENCE.sub("", path.read_text())
    for match in LINK.finditer(text):
        target = match.group(1) or match.group(2)
        if not re.match(r"^[a-z]+:", target):
            yield target


class TestDocumentationPaths(unittest.TestCase):
    def test_readme_example_trains_one_step(self):
        blocks = re.findall(
            r"```python\n(.*?)```", (ROOT / "README.md").read_text(), re.DOTALL
        )
        self.assertEqual(len(blocks), 1)
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "readme_example.py"
            script.write_text(blocks[0])
            namespace = runpy.run_path(str(script))
        self.assertTrue(math.isfinite(namespace["loss"].item()))
        self.assertIsNotNone(namespace["model"].weight.grad)

    def test_relative_links_and_anchors_resolve(self):
        checked = 0
        for page in PUBLISHED:
            for target in relative_links(page):
                with self.subTest(page=str(page.relative_to(ROOT)), target=target):
                    file_part, _, fragment = target.partition("#")
                    destination = (
                        page if not file_part else (page.parent / file_part).resolve()
                    )
                    if destination.is_dir():
                        destination = destination / "index.md"
                    self.assertTrue(destination.exists(), destination)
                    if fragment and destination.suffix == ".md":
                        self.assertIn(fragment, anchors(destination))
                    checked += 1
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()
