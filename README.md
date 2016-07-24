# PyFlux
[![PyFlux](http://pyflux.com/pyflux.png)](http://www.pyflux.com/)

[![Join the chat at https://gitter.im/RJT1990/pyflux](https://badges.gitter.im/RJT1990/pyflux.svg)](https://gitter.im/RJT1990/pyflux?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![PyPI version](https://badge.fury.io/py/pyflux.svg)](https://badge.fury.io/py/pyflux)

__PyFlux__ is an open source time series library for the Python programming language. Built upon the NumPy/SciPy/Pandas libraries, PyFlux allows for easy application of a vast array of time series methods and inference capabilities.

See some examples and documentation at [PyFlux.com](http://www.pyflux.com/). Or see worked-through examples in this [talk](https://github.com/RJT1990/talks/blob/master/PyDataTimeSeriesTalk.ipynb) given to the PyData London Meetup in June 2016.

A __Beta__ release is scheduled for the week commencing 25 July, with Cythonized code for improved performance.

## Models

- [ARIMAX models](http://www.pyflux.com/notebooks/ARIMA.html)
- [GARCH models](http://www.pyflux.com/notebooks/GARCH.html)
- [GAS models](http://www.pyflux.com/notebooks/GAS.html)
- [GAS State Space models](http://www.pyflux.com/notebooks/GASStateSpace.html)
- [GP-NARX models](http://www.pyflux.com/notebooks/GPNARX.html)
- [Gaussian State Space models](http://www.pyflux.com/notebooks/GaussianStateSpace.html)
- [Non-Gaussian State Space models](http://www.pyflux.com/notebooks/NonGaussianStateSpace.html)
- [VAR models](http://www.pyflux.com/notebooks/VAR.html)
 - [Bayesian VAR models](http://www.pyflux.com/notebooks/BayesianVAR.html)

## Inference

- [Black Box Variational Inference](http://www.pyflux.com/notebooks/BBVI.html)
- [Laplace Approximation](http://www.pyflux.com/notebooks/Laplace.html)
- [Maximum Likelihood](http://www.pyflux.com/notebooks/MLE.html) and [Penalized Maximum Likelihood](http://www.pyflux.com/notebooks/PML.html)
- [Metropolis-Hastings](http://www.pyflux.com/notebooks/MetropolisHastings.html)

## Installing PyFlux

```{bash}
pip install pyflux
```

## Python Version

PyFlux is tested on Python 2.7 and 3.5.

## Citation

PyFlux is still alpha software so results should be treated with care, but citations are very welcome:

> Ross Taylor. 2016.
> _PyFlux: An open source time series library for Python_
> http://www.pyflux.com
