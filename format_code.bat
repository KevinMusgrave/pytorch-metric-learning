black . --exclude examples
isort . --profile black --skip-glob examples
nbqa black examples
nbqa isort examples --profile black