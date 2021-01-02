cd src && black pytorch_metric_learning && isort pytorch_metric_learning --profile black
cd .. && black tests && isort tests --profile black