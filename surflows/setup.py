from setuptools import setup, find_packages

setup(
    name='surflows', 
    version='1.0', 
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "pyyaml",
        "tqdm"
    ],
)
