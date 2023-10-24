import io
import os
import re
import sys
import subprocess

from setuptools import setup, find_packages

PACKAGE_NAME = 'pyflux'
DESCRIPTION = "PyFlux: A time-series analysis library for Python"
AUTHOR = "Ross Taylor"
AUTHOR_EMAIL = "rj-taylor@live.co.uk"
URL = 'https://github.com/rjt1990/pyflux'
DOWNLOAD_URL = 'https://github.com/rjt1990/pyflux/tarball/0.4.17'
LICENSE = 'BSD'

# Define Python version requirement
PYTHON_REQUIRES = '>=3.6'

def version(package, encoding='utf-8'):
    """Obtain the package version from a python file e.g. pkg/__init__.py."""
    path = os.path.join(os.path.dirname(__file__), package, '__init__.py')
    with io.open(path, encoding=encoding) as fp:
        version_info = fp.read()
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_info, re.M)
    if not version_match:
        raise RuntimeError(f"Unable to find version string in {path}")
    return version_match.group(1)

def generate_cython(package):
    """Cythonize all sources in the package."""
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    result = subprocess.run([sys.executable,
                            os.path.join(cwd, 'tools', 'cythonize.py'),
                            package],
                        cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError("Running cythonize failed!")

setup(
    name=PACKAGE_NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    version=version(PACKAGE_NAME),
    license=LICENSE,
    python_requires=PYTHON_REQUIRES,  # Specify the required Python version
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scipy', 'numdifftools', 'patsy'],
    keywords=['time series', 'machine learning', 'bayesian statistics'],
)
