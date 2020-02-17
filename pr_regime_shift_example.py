#!/usr/bin/env python
# coding=utf-8
"""
bayesian search for changes in precipitation distribution
"""
import numpy as np
import pymc3 as pm
import pandas as pd
import theano.tensor as tt


class ZeroInflatedGamma(pm.distributions.Continuous):
    """ pymc3 class for Zero Inflated Gamma distribution """

    def __init__(self, alpha, beta, pi, *args, **kwargs):
        super(ZeroInflatedGamma, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.pi = tt.as_tensor_variable(pi)
        self.gamma = pm.distributions.Gamma.dist(alpha, beta)

    def logp(self, value):
        return tt.switch(value > 0, tt.log(1 - self.pi) + self.gamma.logp(value), tt.log(self.pi))


def bayesian_tipping_point(obs_data):
    """

    :param obs_data: 1-d numpy array containing the daily precipitation data
    :return: summary of sampled values and trace itself
    """
    n_dd = obs_data.shape[0]
    idx = np.arange(n_dd)
    with pm.Model() as model:
        alpha_1 = pm.Uniform("alpha_1", lower=0, upper=10)
        alpha_2 = pm.Uniform("alpha_2", lower=0, upper=10)
        beta_1 = pm.Uniform("beta_1", lower=0, upper=10)
        beta_2 = pm.Uniform("beta_2", lower=0, upper=10)
        pi_1 = pm.Uniform("pi_1", lower=0, upper=0.9)
        pi_2 = pm.Uniform("pi_2", lower=0, upper=0.9)
        tau = pm.DiscreteUniform("tau", lower=365 * (5/4.), upper=n_dd - 365 * (5/4.))
        alpha_ = pm.math.switch(tau >= idx, alpha_1, alpha_2)
        beta_ = pm.math.switch(tau >= idx, beta_1, beta_2)
        pi_ = pm.math.switch(tau >= idx, pi_1, pi_2)
        observation = ZeroInflatedGamma("obs", alpha=alpha_, beta=beta_, pi=pi_, observed=obs_data)
        step = pm.NUTS()
        trace = pm.sample(5000,  tune=20000, step=step, nuts_kwargs=dict(target_accept=.9))
        summary = pm.stats.summary(trace)
    return summary, trace