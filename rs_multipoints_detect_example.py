"""
Recursive Bayesian tipping points detection for normally distributed time-series
"""
import numpy as np
import pymc3 as pm
import pandas as pd
import calendar


def bayesian_tipping_point(obs_data):
    """
    :param obs_data: 1-d numpy array containing the daily p recipitation data
    :return: summary of sampled values and trace itself
    """
    obs_std = np.std(obs_data)
    obs_mn = np.mean(obs_data)
    print(obs_mn, obs_std)
    n_dd = obs_data.shape[0]
    idx = np.arange(n_dd)
    with pm.Model() as model:
        mu_1 = pm.Normal("mu_1", mu=obs_mn, sd=obs_std)
        mu_2 = pm.Normal("mu_2", mu=obs_mn, sd=obs_std)
        sd_1 = pm.Normal("sd_1", mu=obs_std, sd=obs_std/3)
        sd_2 = pm.Normal("sd_2", mu=obs_std, sd=obs_std/3)
        tau = pm.DiscreteUniform("tau", lower=84, upper=n_dd - 84)
        mu_ = pm.math.switch(tau >= idx, mu_1, mu_2)
        sd_ = pm.math.switch(tau >= idx, sd_1, sd_2)
        observation = pm.Normal("obs", mu=mu_, sd=sd_, observed=obs_data)
        trace = pm.sample(10000,  tune=40000, njobs=1)
        summary = pm.stats.summary(trace)
    return summary, trace


def get_tp(obs_data, n_years_thold=20, min_len=240):
    """
    the function analyzes trace and summary returned by beysian_tipping_point()
    and return regime shift position or None if no regime shift detected
    :param obs_data:
    :param n_years_thold:
    :param min_len:
    :return: tipping point position, summary, trace
    """
    s, t = None, None
    if len(obs_data) > min_len:
        # detrending might improve results in some applications
        # obs_data = signal.detrend(obs_data)
        try:
            s, t = bayesian_tipping_point(obs_data)
            hst, bins = np.histogram(t['tau'], np.arange(0, len(obs_data), 12))
            n_years = len(bins) - 1
            pd_per_year = np.sum(hst) / n_years
            thold = pd_per_year * n_years_thold
            # here different implementations of threshold is possible
            # for example
            # thold = np.sum(hst) * tau_fraction
            tp_year = hst.argmax()
        except ValueError as e:
            print(e)
            return None
        if hst[tp_year] > thold:
            tp = np.argmax(hst)*12
        else:
            tp = None
    else:
        tp = None
    return tp, s, t


def split(obs_data, dti):
    """
    Recursive function for finding all the regime shifts in the timeseries
    :param obs_data: timeseries as numpy array
    :param dti: time index as pandas.DatetimeIndex or equivalent
    :return: list of indexes where tipping point is detected
    """
    tps = []
    tp_ind, summary, trace = get_tp(obs_data)
    if tp_ind is None:
        return tps
    else:
        obs_data1, obs_data2 = obs_data[:tp_ind], obs_data[tp_ind:]
        dti1, dti2 = dti[:tp_ind], dti[tp_ind:]
        tp_inds1 = split(obs_data1, dti1)
        tp_inds2 = split(obs_data2, dti2)
        tps = tp_inds1 + [dti[tp_ind]] + tp_inds2
    return tps



def anoms(dat, y_min=1901, y_max=2000):
    anoms = dat.copy()
    norm_mask = (anoms.index.year >= y_min) & (anoms.index.year <= y_max)
    for m in range(1, 13):
        mask = (anoms.index.month == m)
        anoms[mask] = dat[mask] - dat[mask & norm_mask].mean()
    return anoms


def read_obs(fn, monthly=True):
    dat = pd.read_csv(fn, header=None)
    d = dat[1]
    d.index = pd.DatetimeIndex(dat[0])
    d = d.resample('M').sum()
    if monthly:
        ndays = [calendar.monthrange(dt.year, dt.month)[1] for dt in d.index]
        d = d * pd.Series(ndays, index=d.index)
    d = anoms(d)
    return d


if __name__ == '__main__':
    dat = pd.read_csv(fn) # dat in this example is pandas.Series
    tps = split(dat.values, pd.Index(range(len(dat))))

