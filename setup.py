# -*- coding: utf-8 -*-
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

with open(here / "requirements.txt") as f:
    install_requires = f.read().splitlines()

dev_requires = [
    "black>=22.1.0",
    "importlib-resources>=5.4.0",
    "ipython>=7.31.0",
    "matplotlib>=3.5.1",
    "notebook>=6.4.8",
    "pandas>=1.4.1",
    "pre-commit>=2.17.0",
    "seaborn>=0.11.2",
]

test_requires = []

extras_require = {  # Optional
    "dev": dev_requires,
    "test": test_requires,
}


setup_kwargs = {
    "name": "pumer",
    "version": "0.1.0",
    "description": "",
    "long_description": long_description,
    "author": "Qingqing Cao",
    "author_email": None,
    "maintainer": None,
    "maintainer_email": None,
    "url": None,
    "package_dir": {"": "src"},
    "packages": find_packages("src"),
    "install_requires": install_requires,
    "extras_require": extras_require,
    "python_requires": ">=3.8,<4.0",
}


setup(**setup_kwargs)
