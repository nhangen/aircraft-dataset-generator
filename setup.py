#!/usr/bin/env python3
"""Setup script for Aircraft Dataset Generator"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aircraft-dataset-generator",
    version="0.1.0",
    author="Contributors",
    description="Comprehensive toolkit for generating synthetic aircraft datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nhangen/aircraft-dataset-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "tqdm>=4.62.0",
        "trimesh>=3.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "tigl": [
            # TiGL must be installed via conda: conda install -c dlr-sc tigl=3.3.0
            # This extra just documents the dependency
        ],
        "all": [
            # All optional dependencies
        ],
    },
    entry_points={
        "console_scripts": [
            "aircraft-generate=aircraft_toolkit.cli:main",
        ],
    },
)