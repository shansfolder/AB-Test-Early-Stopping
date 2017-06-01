import csv
import datetime
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pystan import StanModel
from scipy import stats
from scipy.stats import gaussian_kde, cauchy, norm

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
data_before_time = lambda dat, time_idx: dat[dat.time < time_idx]
readData = lambda fileName: pd.read_csv(dataDir + fileName, low_memory=False)


def obrien_fleming(information_fraction, alpha=0.05):
    """
    Calculate an approximation of the O'Brien-Fleming alpha spending function.
    Args:
        information_fraction (scalar or array_like): share of the information 
            amount at the point of evaluation, e.g. the share of the maximum 
            sample size
        alpha: type-I error rate
    Returns:
        float: redistributed alpha value at the time point with the given 
               information fraction
    """
    return (1 - norm.cdf(norm.ppf(1 - alpha / 2) / np.sqrt(information_fraction))) * 2


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
    ctrl = df.loc[df.variant == 'Control', kpi]
    treat = df.loc[df.variant == 'Treatment', kpi]

    ctrl = ctrl.dropna()
    treat = treat.dropna()

    n_c = len(ctrl)
    n_t = len(treat)

    # n_c = sample_size(ctrl)
    # n_t = sample_size(treat)

    fit_data = {'Nc': n_c, 'Nt': n_t, 'x': ctrl, 'y': treat}
    fit = sm.sampling(data=fit_data, iter=25000, chains=4, n_jobs=1)
    traces = fit.extract()
    return fit, traces


def sample_size(x):
    """
    Calculates sample size of a sample x
    Args:
        x (array_like): sample to calculate sample size
    Returns:
        int: sample size of the sample excluding nans
    """
    # cast into a dummy numpy array to infer the dtype
    if ~isinstance(x, np.ndarray):
        dummy = np.array(x)
    is_numeric = np.issubdtype(dummy.dtype, np.number)

    if is_numeric:
        # Coercing missing values to right format
        _x = np.array(x, dtype=float)
        x_nan = np.isnan(_x).sum()

    # assuming categorical sample
    elif isinstance(x, pd.core.series.Series):
        x_nan = x.str.contains('NA').sum()
    else:
        x_nan = list(x).count('NA')

    return len(x) - x_nan


def bayes_factor(dataset, stan_model, kpi, day_index):
    print("day", day_index)
    df = data_before_time(dataset, day_index + 1)

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

    return (day_index,
            bf_01,
            significant_and_stop_bf,
            hdi_width,
            stop_bp,
            mean_delta,
            lower,
            upper,
            significant_based_on_interval)


def group_sequential(dataset, kpi_name, day_index):
    print("day", day_index)

    dataUntilDay = data_before_time(dataset, day_index + 1)

    information_fraction = len(dataUntilDay) / len(dataset)
    alpha_new = obrien_fleming(information_fraction)

    # calculate the z-score bound
    cap = 8
    bound = norm.ppf(1 - alpha_new / 2)
    # replace potential inf with an upper bound
    if bound == np.inf:
        bound = cap
        
        
    ctrl = dataUntilDay.loc[dataUntilDay.variant == 'Control', kpi_name]
    treat = dataUntilDay.loc[dataUntilDay.variant == 'Treatment', kpi_name]
    mu_c = np.nanmean(ctrl)
    mu_t = np.nanmean(treat)
    sigma_c = np.nanstd(ctrl)
    sigma_t = np.nanstd(treat)
    n_c = sample_size(ctrl)
    n_t = sample_size(treat)
    z = (mu_t - mu_c) / np.sqrt(sigma_c ** 2 / n_c + sigma_t ** 2 / n_t)

    if z > bound or z < -bound:
        stop = True
    else:
        stop = False

    mean_delta = mu_t - mu_c

    interval = normal_difference(mu_c,
                                 sigma_c,
                                 n_c,
                                 mu_t,
                                 sigma_t,
                                 n_t,
                                 [alpha_new * 100 / 2, 100 - alpha_new * 100 / 2])

    lower = interval[0][1]
    upper = interval[1][1]
    significant_based_on_interval = 0 < lower or 0 > upper

    return (day_index, stop, mean_delta, significant_based_on_interval)


def printStatisticsOfDatasets():
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


def createTimeIndex(dataframe):
    # create a dict from date to day_index
    date_column = dateColumn(dataframe)
    date_values = np.sort(dataframe[date_column].unique())

    time_index_map = dict([(date, index) for (index, date) in enumerate(date_values)])
    date_idx_column = list(map(lambda x: time_index_map.get(x), dataframe[date_column]))

    return dataframe.assign(time=date_idx_column)


def addDerivedKPIColumn(dataframe, derived_kpi_name, numerator_column, denominator_column):
    ctrl_reference_kpis = dataframe.loc[dataframe.variant == 'Control', denominator_column]
    treat_reference_kpis = dataframe.loc[dataframe.variant == 'Treatment', denominator_column]

    n_nan_ref_ctrl = sum(ctrl_reference_kpis == 0) + np.isnan(ctrl_reference_kpis).sum()
    n_non_nan_ref_ctrl = len(ctrl_reference_kpis) - n_nan_ref_ctrl

    n_nan_ref_treat = sum(treat_reference_kpis == 0) + np.isnan(treat_reference_kpis).sum()
    n_non_nan_ref_treat = len(treat_reference_kpis) - n_nan_ref_treat

    ctrl_weights = n_non_nan_ref_ctrl * ctrl_reference_kpis / np.nansum(ctrl_reference_kpis)
    treat_weights = n_non_nan_ref_treat * treat_reference_kpis / np.nansum(treat_reference_kpis)

    newColumn = {derived_kpi_name: dataframe[numerator_column] / dataframe[denominator_column]}
    dataframe = dataframe.assign(**newColumn)
    dataframe.loc[dataframe.variant == 'Control', derived_kpi_name] *= ctrl_weights
    dataframe.loc[dataframe.variant == 'Treatment', derived_kpi_name] *= treat_weights

    msg = derived_kpi_name + ": " + str(np.isnan(dataframe[derived_kpi_name]).sum()) + \
          " out of " + str(len(dataframe)) + " is nan."
    print(msg)
    return dataframe


def enrichDataset(fileName, derived_kpi_name, numerator_column, denominator_column):
    original_dataset = readData(fileName)
    time_indexed_dataset = createTimeIndex(original_dataset)
    enhanced_dataset = addDerivedKPIColumn(time_indexed_dataset,
                                           derived_kpi_name,
                                           numerator_column,
                                           denominator_column)
    return enhanced_dataset


def runBayes(fileName, derived_kpi_name, numerator_column, denominator_column):
    print(fileName)

    data = enrichDataset(fileName, derived_kpi_name, numerator_column, denominator_column)
    days = len(data.time.unique())
    stan_model = StanModel(file="../model/normal_kpi.stan")

    start_time_bf = datetime.datetime.now()
    results = Parallel(n_jobs=int(4))(
        delayed(bayes_factor)(data,
                              stan_model,
                              derivedKPI,
                              d) for d in range(days))

    end_time_bf = datetime.datetime.now()
    bf_time_used = end_time_bf - start_time_bf
    print("Bayes factor time spent in seconds:" + str(bf_time_used.seconds))

    filename = '../output/bayes_factor' + fileName
    relativeBasedir = os.path.dirname(filename)
    if not os.path.exists(relativeBasedir):
        os.makedirs(relativeBasedir)

    with open(filename, 'w+') as file_handler:
        out = csv.writer(file_handler)
        for item in results:
            out.writerow(item)


def runGroupSequential(fileName, derived_kpi_name, numerator_column, denominator_column):
    print(fileName)

    data = enrichDataset(fileName, derived_kpi_name, numerator_column, denominator_column)
    days = len(data.time.unique())

    start_time_gs = datetime.datetime.now()
    results = Parallel(n_jobs=int(4))(
        delayed(group_sequential)(data, derivedKPI, d) for d in range(days))

    end_time_gs = datetime.datetime.now()
    gs_time_used = end_time_gs - start_time_gs
    print("Group sequential time spent in seconds:" + str(gs_time_used.seconds))

    filename = '../output/group_sequential' + fileName
    relativeBasedir = os.path.dirname(filename)
    if not os.path.exists(relativeBasedir):
        os.makedirs(relativeBasedir)

    with open(filename, 'w+') as file_handler:
        out = csv.writer(file_handler)
        for item in results:
            out.writerow(item)



if __name__ == "__main__":
    printStatisticsOfDatasets()

    fileNames = ["lipstick_catalog_naviTracking_bunchbox_IT_processed.csv"]
    derivedKPI = "CRpS"
    for fileName in fileNames:
        runBayes(fileName, derivedKPI, "orders", "sessions")
        print("---------------------------------")
