import numpy as np
import pandas as pd
from scipy.stats import poisson, uniform, norm, cauchy, gaussian_kde
from pystan import StanModel
import datetime
import matplotlib.pyplot as plt

seed = 7
days = 20
total_entities = 20000  # entity means e.g. person
averageVisitPerEntity = 3
delta = 0.04

testScales = [1.414]   #1.93 = best scale from power analysis
testDays = [5]
testNumberOfIterations= range(1000, 15001, 1000)


def _randomNumberPoisson(lam):
    lower = poisson.pmf(0, lam)
    return poisson.ppf(uniform.rvs(size=1, loc=lower, scale=1 - lower), lam)


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


def fit_stan(sm, df, kpi, n_iter):
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

    fit = sm.sampling(data=fit_data, iter=n_iter, chains=4, n_jobs=1)
    traces = fit.extract()
    return fit, traces


def generateData(seed, scale):
    np.random.seed(seed)
    assignment = pd.DataFrame({'entity': range(total_entities),
                               'variant': np.random.choice(['A', 'B'],
                                                           size=total_entities,
                                                           p=[0.6, 0.4])})
    all_data = pd.DataFrame()
    for e in range(total_entities):
        n_for_e = int(_randomNumberPoisson(averageVisitPerEntity))
        if n_for_e > days:
            n_for_e = days

        # an entity/person visit in these days: timePoints
        timePoints = np.random.choice(days, size=n_for_e, replace=False)

        normal_shifted_rv = norm.rvs(size=n_for_e, loc=0, scale=scale)
        if assignment.variant[assignment.entity == e].iloc[0] == 'B':
            normal_shifted_rv = norm.rvs(size=n_for_e, loc=delta, scale=scale)

        df = pd.DataFrame({
            'entity': e,
            'normal_shifted': normal_shifted_rv,
            'time': timePoints
        })
        all_data = all_data.append(df, ignore_index=True)

    all_data = pd.merge(all_data, assignment, on='entity')
    print("finish generating data with variance %s", str(scale))
    return all_data


def bayes_factor(stan_model, all_data, testDay, kpi, n_iter):
    """
    Args:
        sm (pystan.model.StanModel): precompiled Stan model object
        simulation_index (int): random seed used for the simulation
        day_index (int): time step of the peeking
        kpi (str): KPI name

    Returns:
        Bayes factor based on the Savage-Dickey density ratio
    """

    snapshot = all_data[all_data.time < testDay+1]
    df = snapshot.groupby(['entity', 'variant']).mean().reset_index()

    fit, traces = fit_stan(stan_model, df, kpi, n_iter)
    kde = gaussian_kde(traces['delta'])
    hdi = HDI_from_MCMC(traces['delta'])
    hdi_width = hdi[1] - hdi[0]

    prior = cauchy.pdf(0, loc=0, scale=1)
    bf_01 = kde.evaluate(0)[0]/prior
    mean_delta = np.mean(traces['delta'])
    return bf_01, hdi_width, mean_delta


def plotData(x, y1, y2, y3, title):
    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    y3 = np.asarray(y3)
    plt.plot(x, y1, 'c', label="Bayes factor")
    plt.plot(x, y2, 'r', label="hdi width")
    plt.plot(x, y3, 'm', label="delta mean")
    plt.legend()
    plt.xlabel("number of iteration in MCMC")
    plt.ylabel("value of interest")
    plt.title(title)
    plt.show()
    return



if __name__ == "__main__":
    stan_model = StanModel(file="/Users/shuang/PersWorkSpace/ABTestEarlyStopping/model/normal_kpi.stan")
    start_time = datetime.datetime.now()

    for scale in testScales:
        all_data = generateData(seed, scale)
        for test_day in testDays:
            x = []
            y1 = []
            y2 = []
            y3 = []
            for number_iteration in testNumberOfIterations:
                print("Test with #iter=", number_iteration)
                bf_01, hdi_width, mean_delta = bayes_factor(stan_model, all_data, test_day, "normal_shifted", number_iteration)

                x.append(number_iteration)
                y1.append(bf_01)
                y2.append(hdi_width)
                y3.append(mean_delta)

            variance = scale ** 2
            plt_title = "Test on day " + str(test_day) + " with data variance " + str(variance)
            plotData(x, y1, y2, y3, plt_title)

    end_time = datetime.datetime.now()
    time_used = end_time - start_time
    print("All time spent in seconds:" + str(time_used.seconds))