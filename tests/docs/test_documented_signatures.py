import ast
import importlib
import inspect
import re
import unittest
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"
MODULES = ("distances", "losses", "miners", "reducers", "regularizers", "samplers")
BLOCK = re.compile(
    r"```python\n((?:" + "|".join(MODULES) + r")\.(\w+)\(((?:.|\n)*?)\))\s*:?\s*\n```"
)
MISSING = object()


def documented_signatures():
    for path in sorted(DOCS.glob("*.md")):
        for match in BLOCK.finditer(path.read_text()):
            module_name = match.group(1).split(".")[0]
            yield path.name, module_name, match.group(2), match.group(3)


def parse_parameters(class_name, arguments, namespace):
    function = ast.parse(f"def {class_name}({arguments}): pass").body[0]
    assert isinstance(function, ast.FunctionDef)
    args = function.args
    positional = args.posonlyargs + args.args
    defaults = [MISSING] * (len(positional) - len(args.defaults)) + list(args.defaults)
    parameters = {}
    for arg, default in zip(positional, defaults, strict=True):
        parameters[arg.arg] = (inspect.Parameter.POSITIONAL_OR_KEYWORD, default)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        parameters[arg.arg] = (inspect.Parameter.KEYWORD_ONLY, default)
    return {
        name: (kind, evaluate_default(default, namespace))
        for name, (kind, default) in parameters.items()
    }


def evaluate_default(default, namespace):
    if not isinstance(default, ast.expr):
        return MISSING
    return eval(compile(ast.Expression(default), "<doc>", "eval"), namespace)


def explicit_parameters(cls):
    return {
        name: parameter
        for name, parameter in inspect.signature(cls).parameters.items()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def accepted_parameters(cls):
    accepted = {}
    for base in reversed(cls.__mro__):
        init = base.__dict__.get("__init__")
        if init is None or base is object:
            continue
        accepted.update(explicit_parameters(init))
    accepted.pop("self", None)
    return accepted


class TestDocumentedSignatures(unittest.TestCase):
    def test_documented_signatures_match_public_classes(self):
        signatures = list(documented_signatures())
        self.assertGreater(len(signatures), 60)
        for page, module_name, class_name, arguments in signatures:
            with self.subTest(page=page, cls=f"{module_name}.{class_name}"):
                module = importlib.import_module(
                    f"pytorch_metric_learning.{module_name}"
                )
                self.assertIn(class_name, module.__all__)
                cls = getattr(module, class_name)
                documented = parse_parameters(class_name, arguments, {class_name: cls})
                own = explicit_parameters(cls)
                accepted = accepted_parameters(cls)
                self.assertLessEqual(set(own), set(documented))
                self.assertLessEqual(set(documented), set(accepted))
                for name, (kind, default) in documented.items():
                    actual = accepted[name]
                    if name in own:
                        self.assertEqual(kind, own[name].kind, name)
                    if actual.default is inspect.Parameter.empty:
                        self.assertIs(default, MISSING, name)
                    elif default is not MISSING:
                        self.assertEqual(default, actual.default, name)


if __name__ == "__main__":
    unittest.main()
