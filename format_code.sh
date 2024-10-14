black src tests
isort src tests --profile black
nbqa black examples
nbqa isort examples --profile black