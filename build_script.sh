./format_code.sh
python -m unittest discover && \
WITH_COLLECT_STATS=true python -m unittest discover && \
rm -rfv build/ && \
rm -rfv dist/ && \
rm -rfv src/pytorch_metric_learning.egg-info/ && \
python3 setup.py sdist bdist_wheel