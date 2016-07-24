
from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'pyflux',
  packages = ['pyflux','pyflux.arma','pyflux.distributions','pyflux.gas','pyflux.garch','pyflux.inference','pyflux.output','pyflux.tests','pyflux.var','pyflux.gpnarx','pyflux.ssm'], 
  version = '0.3.2',
  description = 'A time-series analysis library for Python',
  author = 'Ross Taylor',
  author_email = 'rj-taylor@live.co.uk',
  url = 'https://github.com/rjt1990/pyflux', 
  download_url = 'https://github.com/rjt1990/pyflux/tarball/0.3.2', 
  keywords = ['time series','machine learning','bayesian statistics'],
  license = 'BSD',
  install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn','numdifftools','patsy', 'Cython'],
  ext_modules=cythonize('pyflux.arma.pyx')
)