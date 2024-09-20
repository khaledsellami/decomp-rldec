# setup.py
from setuptools import setup, find_packages
import glob
import os
import re


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open(os.path.join(os.path.dirname(__file__), "rldec", "_version.py")) as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ *= *['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        __version__ = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

if __name__ == "__main__":
    setup(
        name='rldec',
        version=__version__,
        packages=find_packages(exclude=['tests', 'experimenting', 'analysis']),
        install_requires=requirements,
        python_requires=">=3.10",
        package_data={'rldec': ['logging.conf']},
        data_files=[('default_configs', glob.glob(os.path.join("rldec", "default_configs", "")+'*.json'))],
        include_package_data=True,
    )