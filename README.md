# PyFlux
[![PyFlux](http://pyflux.com/pyflux.png)](http://www.pyflux.com/)

[![Join the chat at https://gitter.im/RJT1990/pyflux](https://badges.gitter.im/RJT1990/pyflux.svg)](https://gitter.im/RJT1990/pyflux?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/pyflux.svg)](https://badge.fury.io/py/pyflux)

__PyFlux__ is an open source time series library for Python. The library has a vast array of modern time series models, as well as a flexible array of inference options (frequentist and Bayesian) that can be applied to these models. By combining breadth of models with breadth of inference, PyFlux allows for probabilistic time series modelling.

See some examples, model explanations and documentation at [PyFlux.com](http://www.pyflux.com/). PyFlux is still only alpha software and can become much better and faster with your contributions! - see [here](https://github.com/RJT1990/pyflux/wiki/Contribution-Guidelines) for guidelines on how you can get involved.

A new release is coming in the first week of November! Including faster inference, new distributions, more intuitive API, and new model types!

## Models

- [ARIMA models](http://www.pyflux.com/arima-models)
 - [ARIMAX models](http://www.pyflux.com/arimax-models)
 - [Dynamic Autoregression models](http://www.pyflux.com/dynamic-autoregression-models)
- [GARCH models](http://www.pyflux.com/garch-models)
 - [Beta-t-EGARCH models](http://www.pyflux.com/beta-t-egarch)
 - [EGARCH-in-mean models](http://www.pyflux.com/egarch-in-mean)
 - [EGARCH-in-mean regression models](http://www.pyflux.com/egarch-m-regression)
 - [Long Memory EGARCH models](http://www.pyflux.com/long-memory-egarch/)
 - [Skew-t-EGARCH models](http://www.pyflux.com/skew-t-egarch/)
 - [Skew-t-EGARCH-in-mean models](http://www.pyflux.com/skew-t-egarch-in-mean/)
- [GAS models](http://www.pyflux.com/gas-models/)
 - [GASX models](http://www.pyflux.com/gasx-models/)
- [GAS State Space models](http://www.pyflux.com/gas-state-space-models/)
- [GP-NARX models](http://www.pyflux.com/gp-narx/)
- [Gaussian State Space models](http://www.pyflux.com/gaussian-state-space-models/)
- [Non-Gaussian State Space models](http://www.pyflux.com/non-gaussian-state-space-models/)
- [VAR models](http://www.pyflux.com/vector-autoregression)
 - [Bayesian VAR models](http://www.pyflux.com/bayesian-vector-autoregression)

## Inference

- [Black Box Variational Inference](http://www.pyflux.com/black-box-variational-inference/)
- [Laplace Approximation](http://www.pyflux.com/laplace-approximation/)
- [Maximum Likelihood](http://www.pyflux.com/maximum-likelihood/) and [Penalized Maximum Likelihood](http://www.pyflux.com/penalized-maximum-likelihood/)
- [Metropolis-Hastings](http://www.pyflux.com/metropolis-hastings)

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
