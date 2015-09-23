import numpy as np
import scipy.stats
import scipy.signal


def sepectral_centroid(spectrum):
    return np.sum(np.arange(1, len(spectrum) * spectrum)) / np.sum(spectrum)


def spectral_flatness(spectrum):
    return scipy.stats.gmean(spectrum) / np.mean(spectrum)


def spectral_flux(spectrum1, spectrum2):
    def _abs_max_ratio(s):
        return np.abs(s) / np.max(np.abs(s))
    d = _abs_max_ratio(spectrum2) - _abs_max_ratio(spectrum1)
    return 0.5 * np.sum(d + np.abs(d))


def crest_factor(signal):
    return np.max(np.abs(signal)) / np.sqrt(np.mean(signal*signal))


def zero_crossing_count(signal):
    return np.round(0.5 * np.sum(np.abs(np.diff(np.sign(signal)))))


def moments(signal):
    mu = np.mean(signal)
    sigma = signal - mu
    framecount = len(signal)

    # moments:
    ss = sigma*sigma
    mu_variance = np.mean(ss)
    sss = ss * sigma
    mu_skewness = np.mean(sss)
    mu_kurtosis = np.mean(sss * sigma)

    # standardize the moments:
    mu_kurtosis = mu_kurtosis / (mu_variance * mu_variance)
    mu_skewness = mu_skewness / np.sqrt(mu_variance)**3
    mu_variance = mu_variance * framecount / (framecount - 1)

    return mu, mu_variance, mu_skewness, mu_kurtosis
