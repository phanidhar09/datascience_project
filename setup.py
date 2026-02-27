from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-e "):
                requirements.append(".")
            else:
                requirements.append(line)
    return requirements

setup(
    name="DataScience_project",
    version="0.0.1",
    author="Phanidhar Reddy",
    description="A package for data science and machine learning",
    packages=find_packages(),
    install_requires= get_requirements(file_path="requirements.txt"),
)