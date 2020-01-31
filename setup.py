import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-metric-learning",
    version="0.9.69",
    author="Kevin Musgrave",
    author_email="tkm45@cornell.edu",
    description="The easiest way to use deep metric learning in your application. Modular, flexible, and extensible. Written in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KevinMusgrave/pytorch-metric-learning",
    packages=setuptools.find_packages(),
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
)