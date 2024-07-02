from __future__ import annotations, division

import os
from math import atanh, pow
from typing import List, Tuple, Union

import numpy as np
import pymc3 as pm
import scipy.stats as sp
from numpy import tanh
from scipy.stats import norm, t


def compute_correlation(
    x: List[Union[float, int]], y: List[Union[float, int]]
) -> Tuple[float, float]:
    x_is_not_normal = is_not_normal_distribution(x)
    y_is_not_normal = is_not_normal_distribution(y)
    if x_is_not_normal or y_is_not_normal:
        res = sp.spearmanr(x, y)
        return res.correlation, res.pvalue
    return sp.pearsonr(x, y)


def get_correlation_strength(score: float) -> str:
    if abs(score) > 0.5:
        return "HIGH"
    elif abs(score) > 0.3:
        return "MODERATE"
    else:
        return "LOW"


def compute_mean_similarity(
    x: List[Union[float, int]], y: List[Union[float, int]]
) -> Tuple[float, float]:
    x_is_not_normal = is_not_normal_distribution(x)
    y_is_not_normal = is_not_normal_distribution(y)
    if x_is_not_normal or y_is_not_normal:
        res = sp.mannwhitneyu(x, y)
    else:
        res = sp.ttest_ind(x, y)
    return res.statistic, res.pvalue


def is_not_normal_distribution(
    x: List[Union[float, int]], alpha_level: float = 0.02
) -> bool:
    _, p_value = sp.shapiro(x)
    return p_value < alpha_level


# When doing multiple hypothesis tests, the multiple testing problem occurs. The more inferences are made, the more likely erroneous inferences become
# Using bonferroni correction we prevent the error by dividing the alpha value by number of tests e.g. 3 tests (a/3)
# Alternative methods are false discovery rate, or doing all tests at the same time (ANOVA)
def bonferroni_correction(alpha_value: float, num_hypothesis: int) -> float:
    if num_hypothesis <= 1:
        return alpha_value
    else:
        return alpha_value / num_hypothesis


def bayesian_regression(dataframe, predictor_columns, target_column):
    X = dataframe[predictor_columns].astype(float).values
    y = dataframe[target_column].astype(float).values

    # Standardize predictor variables for better convergence
    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

    num_predictors = X.shape[1]

    with pm.Model() as model:
        # Define priors
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=num_predictors)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Define linear model
        mu = alpha + pm.math.dot(X_standardized, beta)

        # Define likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Perform MCMC
        # NOTE: reduce cpu count by 1 to avoid overloading the system
        num_cpus = max(os.cpu_count() - 1, 1)
        trace = pm.sample(2000, tune=1000, cores=num_cpus,
                          return_inferencedata=True)

        r_squared = bayesian_r_squared(trace, X, y)

    return trace, r_squared


def bayesian_r_squared(trace, X, y):
    # Extract samples from InferenceData object
    alpha_samples = trace.posterior["alpha"].values.flatten()
    beta_samples = trace.posterior["beta"].values.reshape(
        -1, trace.posterior["beta"].shape[-1])

    X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

    # Compute the posterior predictive mean
    y_pred = np.mean(alpha_samples[:, None] +
                     np.dot(beta_samples, X_standardized.T), axis=0)

    # Compute the total sum of squares (TSS)
    tss = np.sum((y - y.mean())**2)

    # Compute the residual sum of squares (RSS)
    rss = np.sum((y - y_pred)**2)

    # Compute the Bayesian R-squared
    r_squared = 1 - (rss / tss)

    return r_squared


"""
Functions for calculating the statistical significant differences between two dependent or independent correlation
coefficients.
The Fisher and Steiger method is adopted from the R package http://personality-project.org/r/html/paired.r.html
and is described in detail in the book 'Statistical Methods for Psychology'
The Zou method is adopted from http://seriousstats.wordpress.com/2012/02/05/comparing-correlations/
Credit goes to the authors of above mentioned packages!
Author: Philipp Singer (www.philippsinger.info)
"""


def rz_ci(r, n, conf_level=0.95):
    zr_se = pow(1 / (n - 3), 0.5)
    moe = norm.ppf(1 - (1 - conf_level) / float(2)) * zr_se
    zu = atanh(r) + moe
    zl = atanh(r) - moe
    return tanh((zl, zu))


def rho_rxy_rxz(rxy, rxz, ryz):
    num = (ryz - 1 / 2.0 * rxy * rxz) * (
        1 - pow(rxy, 2) - pow(rxz, 2) - pow(ryz, 2)
    ) + pow(ryz, 3)
    den = (1 - pow(rxy, 2)) * (1 - pow(rxz, 2))
    return num / float(den)


def dependent_corr(xy, xz, yz, n, twotailed=True, conf_level=0.95, method="steiger"):
    """
    Calculates the statistic significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, only works for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    """
    if method == "steiger":
        d = xy - xz
        determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
        av = (xy + xz) / 2
        cube = (1 - yz) * (1 - yz) * (1 - yz)

        t2 = d * np.sqrt(
            (n - 1) * (1 + yz) /
            (((2 * (n - 1) / (n - 3)) * determin + av * av * cube))
        )
        p = 1 - t.cdf(abs(t2), n - 3)

        if twotailed:
            p *= 2

        return t2, p
    elif method == "zou":
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(xz, n, conf_level=conf_level)[0]
        U2 = rz_ci(xz, n, conf_level=conf_level)[1]
        rho_r12_r13 = rho_rxy_rxz(xy, xz, yz)
        lower = (
            xy
            - xz
            - pow(
                (
                    pow((xy - L1), 2)
                    + pow((U2 - xz), 2)
                    - 2 * rho_r12_r13 * (xy - L1) * (U2 - xz)
                ),
                0.5,
            )
        )
        upper = (
            xy
            - xz
            + pow(
                (
                    pow((U1 - xy), 2)
                    + pow((xz - L2), 2)
                    - 2 * rho_r12_r13 * (U1 - xy) * (xz - L2)
                ),
                0.5,
            )
        )
        return lower, upper
    else:
        raise Exception("Wrong method!")


def independent_corr(
    xy, ab, n, n2=None, twotailed=True, conf_level=0.95, method="fisher"
):
    """
    Calculates the statistic significance between two independent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between a and b
    @param n: number of elements in xy
    @param n2: number of elements in ab (if distinct from n)
    @param twotailed: whether to calculate a one or two tailed test, only works for 'fisher' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'fisher' or 'zou'
    @return: z and p-val
    """

    if method == "fisher":
        xy_z = 0.5 * np.log((1 + xy) / (1 - (xy if xy != 1.0 else 0.99)))
        ab_z = 0.5 * np.log((1 + ab) / (1 - (ab if ab != 1.0 else 0.99)))
        if n2 is None:
            n2 = n

        se_diff_r = np.sqrt(1 / (n - 3) + 1 / (n2 - 3))
        diff = xy_z - ab_z
        z = abs(diff / se_diff_r)
        p = 1 - norm.cdf(z)
        if twotailed:
            p *= 2

        return z, p
    elif method == "zou":
        L1 = rz_ci(xy, n, conf_level=conf_level)[0]
        U1 = rz_ci(xy, n, conf_level=conf_level)[1]
        L2 = rz_ci(ab, n2, conf_level=conf_level)[0]
        U2 = rz_ci(ab, n2, conf_level=conf_level)[1]
        lower = xy - ab - pow((pow((xy - L1), 2) + pow((U2 - ab), 2)), 0.5)
        upper = xy - ab + pow((pow((U1 - xy), 2) + pow((ab - L2), 2)), 0.5)
        return lower, upper
    else:
        raise Exception("Wrong method!")
