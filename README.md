# PyFlux
[![PyFlux](http://pyflux.com/pyflux.png)](http://www.pyflux.com/)

[![Join the chat at https://gitter.im/RJT1990/pyflux](https://badges.gitter.im/RJT1990/pyflux.svg)](https://gitter.im/RJT1990/pyflux?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/pyflux.svg)](https://badge.fury.io/py/pyflux)
[![Documentation Status](https://readthedocs.org/projects/pyflux/badge/?version=latest)](http://pyflux.readthedocs.io/en/latest/?badge=latest)

__PyFlux__ is an open source time series library for Python. The library has a good array of modern time series models, as well as a flexible array of inference options (frequentist and Bayesian) that can be applied to these models. By combining breadth of models with breadth of inference, PyFlux allows for a probabilistic approach to time series modelling.

See some examples and documentation below. PyFlux is still only alpha software; this means you use it at your own risk, and also that some modules are still in need of being optimized.

[Click here for a getting started guide](http://www.pyflux.com/docs/getting_started.html).

## Models

- [ARIMA models](http://www.pyflux.com/docs/arima.html)
 - [ARIMAX models](http://www.pyflux.com/docs/arimax.html)
 - [Dynamic Autoregression models](http://www.pyflux.com/docs/dar.html)
- [Dynamic Paired Comparison models](http://www.pyflux.com/docs/gas_rank.html)
- [GARCH models](http://www.pyflux.com/docs/garch.html)
 - [Beta-t-EGARCH models](http://www.pyflux.com/docs/egarch.html)
 - [EGARCH-in-mean models](http://www.pyflux.com/docs/egarchm.html)
 - [EGARCH-in-mean regression models](http://www.pyflux.com/docs/egarchmreg.html)
 - [Long Memory EGARCH models](http://www.pyflux.com/docs/lmegarch.html)
 - [Skew-t-EGARCH models](http://www.pyflux.com/docs/segarch.html)
 - [Skew-t-EGARCH-in-mean models](http://www.pyflux.com/docs/segarchm.html)
- [GAS models](http://www.pyflux.com/docs/gas.html)
 - [GASX models](http://www.pyflux.com/docs/gasx.html)
- [GAS State Space models](http://www.pyflux.com/docs/gasllm.html)
- [Gaussian State Space models](http://www.pyflux.com/docs/llm.html)
- [Non-Gaussian State Space models](http://www.pyflux.com/docs/nllm.html)
- [VAR models](http://www.pyflux.com/docs/var.html)

## Inference

- [Black Box Variational Inference](http://www.pyflux.com/docs/bayes.html)
- [Laplace Approximation](http://www.pyflux.com/docs/bayes.html)
- [Maximum Likelihood](http://www.pyflux.com/docs/classical.html) and [Penalized Maximum Likelihood](http://www.pyflux.com/docs/bayes.html)
- [Metropolis-Hastings](http://www.pyflux.com/docs/bayes.html)

## Tools

- [Aggregating Algorithms](http://www.pyflux.com/aggregating-algorithms/)

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
> http://www.pyflux.com

## Supporters

[![ALPIMA](http://www.pyflux.com/supporters/alpima.png)](http://www.alpima.net/)
