from setuptools import setup  # type: ignore

setup(
    name="robax",
    version="0.1",
    author="Michael Lutz",
    description="Robax is a Python package for training modern robotics models.",
    packages=["robax"],
    python_requires=">=3.11",
    install_requires=open("requirements.txt").read().splitlines(),
)
