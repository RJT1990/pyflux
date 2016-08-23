# PyFlux
[![PyFlux](http://pyflux.com/pyflux.png)](http://www.pyflux.com/)

[![Join the chat at https://gitter.im/RJT1990/pyflux](https://badges.gitter.im/RJT1990/pyflux.svg)](https://gitter.im/RJT1990/pyflux?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/pyflux.svg)](https://badge.fury.io/py/pyflux)

__PyFlux__ is an open source time series library for the Python programming language. Built upon the NumPy/SciPy/Pandas libraries, PyFlux allows for easy application of a vast array of time series methods and inference capabilities.

See some examples and documentation at [PyFlux.com](http://www.pyflux.com/). PyFlux is still only alpha software and can become much better with your contributions! - see [here](https://github.com/RJT1990/pyflux/wiki/Contribution-Guidelines) for guidelines.

## Models

- [ARIMA models](http://www.pyflux.com/arima-models)
 - [ARIMAX models](http://www.pyflux.com/arimax-models)
- [GARCH models](http://www.pyflux.com/garch-models)
 - [Beta-t-EGARCH models](http://www.pyflux.com/beta-t-egarch)
 - [EGARCH-in-mean models](http://www.pyflux.com/egarch-in-mean)
 - [EGARCH-in-mean regression models](http://www.pyflux.com/egarch-m-regression)
 - [Skew-t-EGARCH models](http://www.pyflux.com/skew-t-egarch/)
 - [Skew-t-EGARCH-in-mean models](http://www.pyflux.com/skew-t-egarch-in-mean/)
- [GAS models](http://www.pyflux.com/gas-models/)
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

## Installing PyFlux

```{bash}
pip install pyflux
```

## Python Version

Supported on Python 2.7 and 3.5.


## Talks

- [PyData San Francisco 2016](https://github.com/RJT1990/PyData2016-SanFrancisco) - August 2016 - an overview of the GAS side of the library, with a fun application to NFL prediction
- [PyData London Meetup](https://github.com/RJT1990/talks/blob/master/PyDataTimeSeriesTalk.ipynb) - June 2016 - an introduction to the library in its early stages

Or see worked-through examples in this [talk](https://github.com/RJT1990/talks/blob/master/PyDataTimeSeriesTalk.ipynb) given to the PyData London Meetup in June 2016.

## Citation

PyFlux is still alpha software so results should be treated with care, but citations are very welcome:

> Ross Taylor. 2016.
> _PyFlux: An open source time series library for Python_
> http://www.pyflux.com
