from distutils.core import setup
setup(
  name = 'pyflux',
  packages = ['pyflux'], 
  version = '0.1',
  description = 'A time-series analysis library for Python',
  author = 'Ross Taylor',
  author_email = 'rj-taylor@live.co.uk',
  url = 'https://github.com/rjt1990/pyflux', 
  download_url = 'https://github.com/rjt1990/pyflux/tarball/0.1', 
  keywords = ['time series','machine learning','bayesian statistics'],
  license = 'BSD',
  install_req=['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn']
)