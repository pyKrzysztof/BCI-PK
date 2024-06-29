from setuptools import setup, find_packages


setup(
    name="mybci",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "brainflow>=5.12",
        "tensorflow>=2.15",
        "keras>=3.3",
        "numpy",
        "pandas"
    ],
)
