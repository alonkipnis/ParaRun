import numpy as np
import pandas as pd
import TwoSampleHC
import logging

from scipy.stats import poisson, norm, chisquare, binom, chi2_contingency
from scipy.spatial.distance import cosine

import sys
import os

from TwoSampleHC import (HC, binom_test_two_sided_random, two_sample_pvals, 
    binom_test_two_sided, poisson_test_random, binom_var_test_df, binom_var_test)


def two_sample_chi_square(c1, c2, lambda_="pearson"):
    """returns the Chi-Square score of the two samples c1 and c2
     (representing counts). Null cells are ignored. 

    Args: 
     c1, c2 : list of integers
        representing two 1-way contingency tables
     lambda_ : one of :
            "pearson"             1     Pearson's chi-squared statistic.
                                        In this case, the function is
                                        equivalent to `stats.chisquare`.
            "log-likelihood"      0     Log-likelihood ratio. Also known as
                                        the G-test [Rf6c2a1ea428c-3]_.
            "freeman-tukey"      -1/2   Freeman-Tukey statistic.
            "mod-log-likelihood" -1     Modified log-likelihood ratio.
            "neyman"             -2     Neyman's statistic.
            "cressie-read"        2/3   
    
    Returns
    -------
    chisq : score 
        score divided by degree of freedom. 
        this normalization is useful in comparing multiple
        chi-squared scores. See Ch. 9.6.2 in 
        Yvonne M. M. Bishop, Stephen E. Fienberg, and Paul 
        W. Holland ``Discrete Multivariate Analysis''  
    log_pval : log of p-value
    """
    
    if (sum(c1) == 0) or (sum(c2) == 0) :
        return np.nan, 1
    else :
        obs = np.array([c1, c2])
        if lambda_ in ['mod-log-likelihood',
                         'freeman-tukey',
                          'neyman'] :
            obs_nz = obs[:, (obs[0]!=0) & (obs[1]!=0)]
        else :
            obs_nz = obs[:, (obs[0]!=0) | (obs[1]!=0)]

        chisq, pval, dof, exp = chi2_contingency(
                                    obs_nz, lambda_=lambda_)
        if pval == 0:
            Lpval = -np.inf
        else :
            Lpval = np.log(pval)
        return chisq / dof, Lpval
        

def cosine_sim(c1, c2):
    """
    returns the cosine similarity of the two sequences
    (c1 and c2 are assumed to be numpy arrays of equal length)
    """
    return cosine(c1, c2)


def sample_from_mixture(lmd0, lmd1, eps) :
    N = len(lmd0)
    idcs = np.random.rand(N) < eps
    #idcs = np.random.choice(np.arange(N), k)
    lmd = np.array(lmd0.copy())
    lmd[idcs] = np.array(lmd1)[idcs]
    return np.random.poisson(lam=lmd)

def power_law(n, xi) :
    p = np.arange(1.,n+1) ** (-xi)
    return p / p.sum()

def evaluate_iteration(n, N, ep, mu, xi, metric = 'Hellinger') :
    logging.debug(f"Evaluating with: n={n}, N={N}, ep={ep}, mu={mu}, xi={xi}")
    P = power_law(N, xi)
    
    if metric == 'Hellinger' :
      QP = (np.sqrt(P) + np.sqrt(mu))**2

    if metric == 'ChiSq' :
      QP = P + 2 * np.sqrt(P * mu)

    if metric == 'proportional' :
      QP = P *( 1 + r * np.log(N))

    if metric == 'power' :
      QP = P * (np.log(N) ** r)

    smp1 = sample_from_mixture(n*P, n*QP, ep)
    smp2 = sample_from_mixture(n*P, n*QP, ep)

    stbl = False
    gamma = 0.25

    def filter_pvals(pv) :
        return pv[(smp1 == 0) | (smp2 == 0)]

    def test_stats(pv) :
        if len(pv) > 0 :
            hc, _ = HC(pv[pv < 1], stbl=stbl).HC(gamma=gamma)
            min_pv = pv.min()
        else :
            hc = np.nan
            min_pv = np.nan
        return hc, min_pv
    
    pv_rand = two_sample_pvals(smp1, smp2, randomize=True, sym=True)
    pv = two_sample_pvals(smp1, smp2, randomize=False)
    pv_one_NR = poisson_test_random(smp1, n*P)

    # two sample non-random
    hc_rand, MinPv_rand = test_stats(filter_pvals(pv_rand))
    hc, MinPv = test_stats(filter_pvals(pv))
    hc_one_NR, MinPv_one_NR = test_stats(filter_pvals(pv_one_NR))

    chisq, _ = two_sample_chi_square(smp1, smp2)

    cos = cosine_sim(smp1, smp2)

    pv_stripes = binom_var_test(smp1, smp2, sym=True, singleton=False).values
    hc_stripes, MinPv_stripes = test_stats(pv_stripes)

    return { 'HC_random' : hc_rand,
            'minPv_random' : MinPv_rand,
            'HC' : hc,
         'minPv' : MinPv,
         'HC_one_NR' : hc_one_NR,
         'minPv_one_NR' : MinPv_one_NR,
         'chisq' : chisq,
         'cos' : cos,
         'HC_stripes' : hc_stripes,
         'minPv_stripes' : MinPv_stripes
         }
