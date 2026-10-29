import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
PACKAGE = ROOT / "src" / "pytorch_metric_learning"
PUBLIC_MODULES = {
    "distances": ("distances", ["distances.md"]),
    "losses": ("losses", ["losses/*.md"]),
    "miners": ("miners", ["miners.md"]),
    "reducers": ("reducers", ["reducers.md"]),
    "regularizers": ("regularizers", ["regularizers.md"]),
    "samplers": ("samplers", ["samplers.md"]),
    "evaluation": ("evaluation", ["accuracy_calculation.md", "retrieval.md"]),
    "distributed": ("distributed", ["distributed.md"]),
    "types": ("types", ["types.md"]),
    "utils.indexing": ("indexing", ["indexing.md"]),
}
SIGNATURE = re.compile(
    r"^```python \{\.signature\}\n(.*?)^```", re.DOTALL | re.MULTILINE
)
HEADING = re.compile(r"^#{1,3} (\w+)$", re.MULTILINE)


def defined_names(path):
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.TypeAlias):
            names.add(node.name.id)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def source_files(module):
    path = PACKAGE.joinpath(*module.split("."))
    if path.is_dir():
        return [p for p in sorted(path.glob("*.py")) if not p.name.startswith("_")]
    return [path.with_suffix(".py")]


def documented_names(alias, patterns):
    names = set()
    for pattern in patterns:
        for page in sorted(DOCS.glob(pattern)):
            text = page.read_text()
            names.update(HEADING.findall(text))
            for block in SIGNATURE.findall(text):
                match = re.match(
                    rf"(?:{alias}\.(\w+)\(|class (\w+)\(|type (\w+) =)", block
                )
                if match:
                    names.add(next(group for group in match.groups() if group))
    return names


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_public_module_exports_resolve(module):
    imported = importlib.import_module(f"pytorch_metric_learning.{module}")
    exports = imported.__all__
    assert len(exports) == len(set(exports))
    for name in exports:
        assert hasattr(imported, name), name


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_unexported_definitions_are_private(module):
    exports = set(importlib.import_module(f"pytorch_metric_learning.{module}").__all__)
    for path in source_files(module):
        unexported = defined_names(path) - exports
        assert not unexported, f"{path.relative_to(ROOT)}: {unexported}"


@pytest.mark.parametrize("module", PUBLIC_MODULES)
def test_every_export_has_a_reference_entry(module):
    alias, patterns = PUBLIC_MODULES[module]
    exports = set(importlib.import_module(f"pytorch_metric_learning.{module}").__all__)
    assert exports <= documented_names(alias, patterns)


def test_versioning_lists_the_public_modules():
    text = (DOCS / "versioning.md").read_text()
    rule = text.split("## Public API", 1)[1].split("\n\n", 2)[1]
    listed = set(re.findall(r"`([\w.]+)`", rule)) - {
        "__all__",
        "pytorch_metric_learning",
    }
    assert listed == set(PUBLIC_MODULES)


def test_types_page_matches_source():
    source = {
        node.name.id if isinstance(node, ast.TypeAlias) else node.name: ast.dump(node)
        for node in ast.parse((PACKAGE / "types.py").read_text()).body
        if isinstance(node, (ast.TypeAlias, ast.ClassDef))
    }
    documented = {}
    for block in SIGNATURE.findall((DOCS / "types.md").read_text()):
        for node in ast.parse(block).body:
            name = node.name.id if isinstance(node, ast.TypeAlias) else node.name
            documented[name] = ast.dump(node)
    assert documented == source
