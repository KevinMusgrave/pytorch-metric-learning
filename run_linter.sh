flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --select=F401 --statistics --per-file-ignores="__init__.py:F401"
nbqa flake8 examples --count --select=E9,F63,F7,F82 --show-source --statistics
nbqa flake8 examples --count --select=F401 --statistics --per-file-ignores="__init__.py:F401"