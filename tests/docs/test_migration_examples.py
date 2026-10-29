import re
import runpy
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "migration-2-to-3.md"
EXAMPLES = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.DOTALL)


@pytest.mark.parametrize(
    "source", EXAMPLES, ids=[f"example-{i + 1}" for i in range(len(EXAMPLES))]
)
def test_migration_example(source, tmp_path):
    script = tmp_path / "migration_example.py"
    script.write_text(source)
    runpy.run_path(str(script))
