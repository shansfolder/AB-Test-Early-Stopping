import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde, cauchy

dataDir = "../data/real/processed/"
fileNames = ["DTP_CH_web_processed.csv",
             "Editorial_Assortment_Entries_treatment1_processed.csv",
             "Editorial_Assortment_Entries_treatment2_processed.csv",
             "Editorial_Catalog_entries_Msite_processed.csv",
             "lipstick_catalog_naviTracking_bunchbox_IT_processed.csv",
             "lipstick_catalog_naviTracking_bunchbox_NL_processed.csv",
             "segmented_sorting_fasion_floor_fashion_processed.csv",
             "segmented_sorting_fasion_floor_modern_processed.csv",
             "segmented_sorting_fasion_floor_no-floor_processed.csv",
             "segmented_sorting_fasion_floor_trend_processed.csv"]

dateColumn = lambda dat: "date" if "date" in dat.columns else "DATE"


def readData(fileName):
    file = dataDir + fileName
    dat = pd.read_csv(file, low_memory=False)
    return dat


def pooled_std(std1, n1, std2, n2):
    """
    Returns the pooled estimate of standard deviation. Assumes that population
    variances are equal (std(v1)**2==std(v2)**2) - this assumption is checked
    for reasonableness and an exception is raised if this is strongly violated.
    Args:
        std1 (float): standard deviation of first sample
        n1 (integer): size of first sample
        std2 (float): standard deviation of second sample
        n2 (integer): size of second sample
    Returns:
        float: Pooled standard deviation
    For further information visit:
        http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals5.html
    Todo:
        Also implement a version for unequal variances.
    """
    return np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))


def normal_difference(mean1, std1, n1, mean2, std2, n2, percentiles=[2.5, 97.5],
                      relative=False):
    """
    Calculates the difference distribution of two normal distributions.
    Computation is done in form of treatment minus control. It is assumed that
    the standard deviations of both distributions do not differ too much.
    Args:
        mean1 (float): mean value of the treatment distribution
        std1 (float): standard deviation of the treatment distribution
        n1 (integer): number of samples of the treatment distribution
        mean2 (float): mean value of the control distribution
        std2 (float): standard deviation of the control distribution
        n2 (integer): number of samples of the control distribution
        percentiles (list): list of percentile values to compute
        relative (boolean): If relative==True, then the values will be returned
            as distances below and above the mean, respectively, rather than the
            absolute values. In	this case, the interval is mean-ret_val[0] to
            mean+ret_val[1]. This is more useful in many situations because it
            corresponds with the sem() and std() functions.
    Returns:
        dict: percentiles and corresponding values
    For further information vistit:
            http://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals5.html
    """
    # Compute combined parameters from individual parameters
    mean = mean1 - mean2
    std = pooled_std(std1, n1, std2, n2)
    # Computing standard error
    st_error = std * np.sqrt(1. / n1 + 1. / n2)
    # Computing degrees of freedom
    d_free = n1 + n2 - 2

    # Mapping percentiles via standard error
    if relative:
        return list([(p, stats.t.ppf(p / 100.0, df=d_free) * st_error)
                     for p in percentiles])
    else:
        return list([(p, mean + stats.t.ppf(p / 100.0, df=d_free) * st_error)
                     for p in percentiles])


def HDI_from_MCMC(posterior_samples, credible_mass=0.95):
    # Computes highest density interval from a sample of representative values,
    # estimated as the shortest credible interval
    # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
    # http://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [0] * nCIs
    for i in range(0, nCIs):
        ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth)) + ciIdxInc]
    return (HDImin, HDImax)


def fit_stan(sm, df, kpi):
    """
    Args:
        sm (pystan.model.StanModel): precompiled Stan model object
        fit_data (dict): mapping between the observed data and the variable name

    Returns:
        fit (StanFit4Model)
        delta_trace (array)
    """
    if 'normal' in kpi:
        fit_data = {'Nc': sum(df.variant == 'A'),
                    'Nt': sum(df.variant == 'B'),
                    'x': df[kpi][df.variant == 'A'],
                    'y': df[kpi][df.variant == 'B']}
    else:
        raise NotImplementedError

    fit = sm.sampling(data=fit_data, iter=15000, chains=4, n_jobs=1)

    # extract the traces
    traces = fit.extract()
    return fit, traces


def bayes_factor(stan_model, kpi):
    """
    Args:
        sm (pystan.model.StanModel): precompiled Stan model object
        simulation_index (int): random seed used for the simulation
        day_index (int): time step of the peeking
        kpi (str): KPI name

    Returns:
        Bayes factor based on the Savage-Dickey density ratio
    """
    print("simulation:" + str(simulation_index) + ", day:" + str(day_index))
    dat = readSimulationData(simulation_index)
    df = get_snapshot(dat, day_index + 1)

    fit, traces = fit_stan(stan_model, df, kpi)
    kde = gaussian_kde(traces['delta'])
    hdi = HDI_from_MCMC(traces['delta'])
    upper = hdi[1]
    lower = hdi[0]
    prior = cauchy.pdf(0, loc=0, scale=1)

    bf_01 = kde.evaluate(0)[0] / prior
    hdi_width = upper - lower
    mean_delta = np.mean(traces['delta'])

    significant_and_stop_bf = bf_01 < 1 / 3.
    stop_bp = hdi_width < 0.08
    significant_based_on_interval = 0 < lower or 0 > upper

    return (simulation_index,
            day_index,
            bf_01,
            significant_and_stop_bf,
            hdi_width,
            stop_bp,
            mean_delta,
            lower,
            upper,
            significant_based_on_interval)


def group_sequential(kpi_name):
    dat = readSimulationData(simulation_index)
    kpi = get_snapshot(dat, day_index + 1)
    ctrl = kpi.loc[kpi.variant == 'A', kpi_name]
    treat = kpi.loc[kpi.variant == 'B', kpi_name]
    mu_c = ctrl.mean()
    mu_t = treat.mean()
    sigma_c = ctrl.std()
    sigma_t = treat.std()
    n_c = len(ctrl)
    n_t = len(treat)
    z = (mu_t - mu_c) / np.sqrt(sigma_c ** 2 / n_c + sigma_t ** 2 / n_t)

    if z > bounds[day_index] or z < -bounds[day_index]:
        stop = True
    else:
        stop = False

    mean_delta = mu_t - mu_c

    interval = normal_difference(mu_c, sigma_c, n_c, mu_t, sigma_t, n_t,
                                 [alpha_new[day_index] * 100 / 2, 100 - alpha_new[day_index] * 100 / 2])

    lower = interval[0][1]
    upper = interval[1][1]
    significant_based_on_interval = 0 < lower or 0 > upper

    return (simulation_index, day_index, stop, mean_delta, significant_based_on_interval)



if __name__ == "__main__":
    for fileName in fileNames:
        print("data file:", fileName)
        dat = readData(fileName)
        date_column = dateColumn(dat)
        print("number of days", len(dat[date_column].unique()))
        print("number of samples", len(dat))
        print("number of control", len(dat[dat.variant == 'Control']))
        print("number of treatment", len(dat[dat.variant == 'Treatment']))
        print("columns", dat.columns)
        print("---------------------------------")
    print("---------------------------------")
