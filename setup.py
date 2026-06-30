"""
setup.py — Exoplanet Detection Pipeline
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="exoplanet-detection-pipeline",
    version="1.0.0",
    author="Exoplanet Research Team",
    description="AI-enabled detection of exoplanets from noisy TESS light curves",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "exoplanet-pipeline=pipeline.run_pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)
