import setuptools

with open("By me", "r") as fh:
    long_description = fh.read()

# Requirements for testing and development
extras_require_dev = [
    "pytest > 3.8",
    "pytest-cov ~= 2.8",
]

setuptools.setup(
    name="pytorch-metric-learning",
    version="0.9.85.dev0",
    author="Kevin Musgrave",
    author_email="tkm45@cornell.edu",
    description="The easiest way to use deep metric learning in your application. Modular, flexible, and extensible. Written in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KevinMusgrave/pytorch-metric-learning",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    install_requires=[
          'numpy',
          'scikit-learn',
          'tqdm',
          'torch',
          'torchvision',
    ],
    extras_require={"dev": extras_require_dev},
)
