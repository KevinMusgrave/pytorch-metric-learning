import re
import runpy
from pathlib import Path

import pytest

PAGES = sorted((Path(__file__).resolve().parents[2] / "docs" / "losses").glob("*.md"))
EXAMPLE = re.compile(
    r"^\*\*Example\*\*:\n\n```python\n(.*?)^```", re.DOTALL | re.MULTILINE
)
WITHOUT_EXAMPLE = {"base_proxy_loss"}


def test_exempt_pages_exist():
    assert {page.stem for page in PAGES} > WITHOUT_EXAMPLE


@pytest.mark.parametrize("page", PAGES, ids=[page.stem for page in PAGES])
def test_loss_page_example(page, tmp_path, monkeypatch):
    text = page.read_text()
    found = EXAMPLE.findall(text)
    expected = 0 if page.stem in WITHOUT_EXAMPLE else 1
    assert text.count("**Example**:") == len(found) == expected
    monkeypatch.chdir(tmp_path)
    for source in found:
        script = tmp_path / f"{page.stem}_example.py"
        script.write_text(source)
        runpy.run_path(str(script))
