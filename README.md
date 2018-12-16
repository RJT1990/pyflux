# PyFlux

[![Join the chat at https://gitter.im/RJT1990/pyflux](https://badges.gitter.im/RJT1990/pyflux.svg)](https://gitter.im/RJT1990/pyflux?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/pyflux.svg)](https://badge.fury.io/py/pyflux)
[![Documentation Status](https://readthedocs.org/projects/pyflux/badge/?version=latest)](http://pyflux.readthedocs.io/en/latest/?badge=latest)

__PyFlux__ is an open source time series library for Python. The library has a good array of modern time series models, as well as a flexible array of inference options (frequentist and Bayesian) that can be applied to these models. By combining breadth of models with breadth of inference, PyFlux allows for a probabilistic approach to time series modelling.

See some examples and documentation below. PyFlux is still only alpha software; this means you use it at your own risk, that test coverage is still in need of expansion, and also that some modules are still in need of being optimized.

[Click here for a getting started guide](http://pyflux.readthedocs.io/en/latest/getting_started.html).

**Note From Author** : I am currently working on other projects as of now, so have paused updates for this library for the immediate future. If you'd like to help move the library forward by contributing, then do get in touch! I am planning to review at end of year and update the library as required (new version requirements, etc).

## Models

- [ARIMA models](http://pyflux.readthedocs.io/en/latest/arima.html)
  - [ARIMAX models](http://pyflux.readthedocs.io/en/latest/arimax.html)
  - [Dynamic Autoregression models](http://pyflux.readthedocs.io/en/latest/docs/dar.html)
- [Dynamic Paired Comparison models](http://pyflux.readthedocs.io/en/latest/gas_rank.html)
- [GARCH models](http://pyflux.readthedocs.io/en/latest/garch.html)
  - [Beta-t-EGARCH models](http://pyflux.readthedocs.io/en/latest/egarch.html)
  - [EGARCH-in-mean models](http://pyflux.readthedocs.io/en/latest/egarchm.html)
  - [EGARCH-in-mean regression models](http://pyflux.readthedocs.io/en/latest/egarchmreg.html)
  - [Long Memory EGARCH models](http://pyflux.readthedocs.io/en/latest/lmegarch.html)
  - [Skew-t-EGARCH models](http://pyflux.readthedocs.io/en/latest/segarch.html)
  - [Skew-t-EGARCH-in-mean models](http://pyflux.readthedocs.io/en/latest/segarchm.html)
- [GAS models](http://pyflux.readthedocs.io/en/latest/gas.html)
  - [GASX models](http://pyflux.readthedocs.io/en/latest/gasx.html)
- [GAS State Space models](http://pyflux.readthedocs.io/en/latest/gas_llm.html)
- [Gaussian State Space models](http://pyflux.readthedocs.io/en/latest/llm.html)
- [Non-Gaussian State Space models](http://pyflux.readthedocs.io/en/latest/nllm.html)
- [VAR models](http://pyflux.readthedocs.io/en/latest/var.html)

## Inference

- [Black Box Variational Inference](http://pyflux.readthedocs.io/en/latest/bayes.html)
- [Laplace Approximation](http://pyflux.readthedocs.io/en/latest/bayes.html)
- [Maximum Likelihood](http://pyflux.readthedocs.io/en/latest/classical.html) and [Penalized Maximum Likelihood](http://pyflux.readthedocs.io/en/latest/bayes.html)
- [Metropolis-Hastings](http://pyflux.readthedocs.io/en/latest/bayes.html)

## Installing PyFlux

```{bash}
pip install pyflux
```

## Python Version

Supported on Python 2.7 and 3.5.

## Talks

- [PyData San Francisco 2016](https://github.com/RJT1990/PyData2016-SanFrancisco) - August 2016 -  a tour of time series (and predicting NFL games)
- [PyData London Meetup](https://github.com/RJT1990/talks/blob/master/PyDataTimeSeriesTalk.ipynb) - June 2016 - an introduction to the library in its early stages

## Citation

PyFlux is still alpha software so results should be treated with care, but citations are very welcome:

> Ross Taylor. 2016.
> _PyFlux: An open source time series library for Python_
