import sys

import setuptools

sys.path.insert(0, "src")
import pytorch_metric_learning

with open("README.md", "r") as fh:
    long_description = fh.read()

extras_require_with_hooks = [
    "record-keeper >= 0.9.32",
    "faiss-gpu >= 1.6.3",
    "tensorboard",
]
extras_require_with_hooks_cpu = [
    "record-keeper >= 0.9.32",
    "faiss-cpu >= 1.6.3",
    "tensorboard",
]
extras_require_docs = ["mkdocs-material"]
extras_require_dev = ["black", "isort", "nbqa", "flake8"]

setuptools.setup(
    name="pytorch-metric-learning",
    version=pytorch_metric_learning.__version__,
    author="Kevin Musgrave",
    description="The easiest way to use deep metric learning in your application. Modular, flexible, and extensible. Written in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KevinMusgrave/pytorch-metric-learning",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    license_files=('LICENSE',),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.0",
    install_requires=[
        "numpy < 2.0",
        "scikit-learn",
        "tqdm",
        "torch >= 1.6.0",
    ],
    extras_require={
        "with-hooks": extras_require_with_hooks,
        "with-hooks-cpu": extras_require_with_hooks_cpu,
        "docs": extras_require_docs,
        "dev": extras_require_dev,
    },
)
