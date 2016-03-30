"""
The module Inference holds estimation procedures.
"""
from priors import Normal, InverseGamma, Uniform
from metropolis_hastings import metropolis_hastings
from norm_post_sim import norm_post_sim
from bbvi import BBVI

