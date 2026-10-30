import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PAGES = [
    ROOT / "README.md",
    *sorted(
        path
        for path in DOCS.rglob("*.md")
        if path.relative_to(DOCS).parts[0] not in ("design", "security")
    ),
]
FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```", re.DOTALL | re.MULTILINE)
RUNNABLE = "python"
SIGNATURE = "python {.signature}"
LOSS_EXAMPLE = re.compile(r"^\*\*Example\*\*:\n\n```python\n", re.MULTILINE)
WITHOUT_LOSS_EXAMPLE = {"base_proxy_loss"}


def page_id(page):
    return str(page.relative_to(ROOT))


def blocks(page, info):
    text = page.read_text()
    for match in FENCE.finditer(text):
        if match.group(1) == info:
            line = text.count("\n", 0, match.start()) + 1
            yield line, match.group(2)


def parse_signature(source):
    try:
        return ast.parse(source)
    except SyntaxError:
        parameters = re.sub(r"^[\w.]+", "", source.strip(), count=1)
        return ast.parse(f"def signature{parameters}: pass")


@pytest.mark.parametrize("page", PAGES, ids=page_id)
def test_python_blocks_are_runnable_or_signatures(page):
    infos = {match.group(1) for match in FENCE.finditer(page.read_text())}
    python_infos = {info for info in infos if info.startswith("py")}
    assert python_infos <= {RUNNABLE, SIGNATURE}


@pytest.mark.parametrize("page", PAGES, ids=page_id)
def test_signature_blocks_parse(page):
    for _, source in blocks(page, SIGNATURE):
        assert parse_signature(source).body


@pytest.mark.parametrize("page", PAGES, ids=page_id)
def test_page_examples_run_in_order(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    for line, source in blocks(page, RUNNABLE):
        exec(compile(source, f"{page_id(page)}:{line}", "exec"), namespace)


def test_runnable_blocks_cover_published_pages():
    runnable = sum(len(list(blocks(page, RUNNABLE))) for page in PAGES)
    signatures = sum(len(list(blocks(page, SIGNATURE))) for page in PAGES)
    assert runnable > 100
    assert signatures > 90


@pytest.mark.parametrize(
    "page",
    sorted((DOCS / "losses").glob("*.md")),
    ids=lambda page: page.stem,
)
def test_loss_page_has_one_example(page):
    text = page.read_text()
    expected = 0 if page.stem in WITHOUT_LOSS_EXAMPLE else 1
    assert text.count("**Example**:") == len(LOSS_EXAMPLE.findall(text)) == expected


def test_pages_without_loss_example_exist():
    assert {page.stem for page in (DOCS / "losses").glob("*.md")} > WITHOUT_LOSS_EXAMPLE
