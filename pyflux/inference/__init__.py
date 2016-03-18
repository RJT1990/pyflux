"""
The module Inference holds estimation procedures.
"""
from priors import Normal, InverseGamma, Uniform
from metropolis_hastings import metropolis_hastings
from laplace import laplace
from hmc import hmc

