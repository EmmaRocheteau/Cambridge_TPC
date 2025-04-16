from setuptools import setup, find_packages

setup(
    name="healthcare_predictions",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "ray[tune]>=2.5.0",
        "optuna>=3.2.0",
        "pandas>=2.0.0",
        "plotly>=5.13.0",
        "pyyaml>=6.0.0",
    ]
)
