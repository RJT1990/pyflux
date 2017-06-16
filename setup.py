import io
import os
import re
import sys
import subprocess

# Use of SETUP built-in adapted from scikit-learn's setup structure.

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins

builtins.__PYFLUX_SETUP__ = True

PACKAGE_NAME = 'pyflux'
DESCRIPTION = "PyFlux: A time-series analysis library for Python"
LONG_DESCRIPTION = DESCRIPTION
AUTHOR = "Ross Taylor"
AUTHOR_EMAIL = "rj-taylor@live.co.uk"
URL = 'https://github.com/rjt1990/pyflux'
DOWNLOAD_URL = 'https://github.com/rjt1990/pyflux/tarball/0.4.15'
LICENSE = 'BSD'

def version(package, encoding='utf-8'):
    """Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    path = os.path.join(os.path.dirname(__file__), package, '__init__.py')
    with io.open(path, encoding=encoding) as fp:
        version_info = fp.read()
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_info, re.M)
    if not version_match:
        raise RuntimeError("Unable to find version string.")
    return version_match.group(1)

def generate_cython(package):
    """Cythonize all sources in the package"""
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'tools', 'cythonize.py'),
                         package],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError("Running cythonize failed!")

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(PACKAGE_NAME)

    return config

def setup_package():
    from numpy.distutils.core import setup

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_path = local_path

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    # Run build
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    cwd = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(cwd, 'PKG-INFO')):
        # Generate Cython sources, unless building from source release
        generate_cython(PACKAGE_NAME)

    try:
        setup(name=PACKAGE_NAME,
              author=AUTHOR,
              author_email=AUTHOR_EMAIL,
              url=URL,
              download_url=DOWNLOAD_URL,
              description=DESCRIPTION,
              long_description = LONG_DESCRIPTION,
              version=version(PACKAGE_NAME),
              license=LICENSE,
              configuration=configuration,
              keywords = ['time series','machine learning','bayesian statistics'],
              install_requires=['numpy', 'pandas', 'scipy', 'numdifftools','patsy'])
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return

if __name__ == '__main__':
    setup_package()