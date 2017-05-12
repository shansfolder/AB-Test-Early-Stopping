from optparse import OptionParser
import numpy as np
import pandas as pd
from pystan import StanModel
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde, cauchy
import datetime
import csv
import os
from expan.core.experiment import Experiment

metadata = {'source': 'simulation', 'experiment': 'random'}

sims = 1#000
days = 20

# dat = pd.read_csv('size39_sales_kpi_over_time.csv')
# dat = generate_random_data()

def get_snapshot(dat, start_time):
    snapshot = dat[dat.time < start_time]
    aggregated = snapshot.groupby(['entity', 'variant']).mean().reset_index()
    return aggregated


def readSimulationData(sim):
    dat = pd.read_csv("data/simulation/simulation" + str(sim) + ".csv")
    return dat


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


# perform the fit
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

    # TODO: fix binomial model
    elif 'binomial' in kpi:
        fit_data = {'Nc': sum(df.variant == 'A'),
                    'Nt': sum(df.variant == 'B'),
                    't_c': df['ARTICLES_ORDERED'][df.variant == 'hide'],
                    't_t': df['ARTICLES_ORDERED'][df.variant == 'show'],
                    's_c': df['ARTICLES_RETURNED'][df.variant == 'hide'].astype(int),
                    's_t': df['ARTICLES_RETURNED'][df.variant == 'show'].astype(int)
                    }
    else:
        raise NotImplementedError

    fit = sm.sampling(data=fit_data, iter=2000, chains=4, n_jobs=1)

    # extract the traces
    traces = fit.extract()
    return fit, traces


def bayes_factor(stan_model, simulation_index, day_index, kpi):
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
    hdi_width = hdi[1] - hdi[0]

    prior = cauchy.pdf(0, loc=0, scale=1)
    return (simulation_index,
            day_index,
            kde.evaluate(0)[0] / prior,
            hdi_width,
            np.mean(traces['delta']))


def fixed_horizon(i):
    """
    Simulate the fixed horizon test for a single run.
    """
    dat = readSimulationData(i)
    # fixed-horizon
    kpi = get_snapshot(dat, days + 1)
    exp = Experiment('A', kpi, metadata)
    res = exp.delta(kpi_subset=['normal_same'])
    interval = res.statistic('delta', 'uplift_pctile', 'normal_same').loc[:, ('value', 'B')]
    if np.sign(interval[0]) * np.sign(interval[1]) > 0:
        diff = True
    else:
        diff = False

    return (interval[0], interval[1], diff)


def group_sequential(simulation_index, day_index, kpi_name):
    # calculated with bounds(seq(0.05,1,length=20),iuse=c(1,1),alpha=c(0.025,0.025)) in R
    bounds = [8.000000, 8.000000, 8.000000, 4.915742, 4.336773, 3.942483, 3.638028, 3.394052, 3.193264, 3.024348,
              2.879692, 2.753971, 2.643399, 2.545164, 2.457130, 2.377652, 2.305414, 2.239395, 2.178743, 2.122766]

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
        diff = True
    else:
        diff = False

    return (simulation_index, day_index, diff, mu_t - mu_c)


def run(func, cpus, **kwargs):
    """
    Wrapper to run different simulations and write results to file.
    """
    start_time = datetime.datetime.now()
    timestamp = '{:%Y%m%d}'.format(start_time)

    if func.__name__ == 'bayes_factor':
        stan_model = StanModel(file=kwargs['model_file'])
        start_time_bf = datetime.datetime.now()
        results = Parallel(n_jobs=int(cpus))(
            delayed(func)(stan_model,
                          s,
                          d,
                          kwargs['kpi'])
            for s in range(sims) for d in range(days)
        )
        filename = func.__name__ + '/' + func.__name__ + '_' + kwargs['kpi'] + '_' + timestamp + '.csv'
        end_time_bf = datetime.datetime.now()
        bf_time_used = end_time_bf - start_time_bf
        print("Bayes factor time spent in seconds:" + str(bf_time_used.seconds))
    elif func.__name__ == 'fixed_horizon':
        results = Parallel(n_jobs=int(cpus))(
            delayed(func)(s) for s in range(sims)
        )
        filename = func.__name__ + '/' + func.__name__ + '_' + timestamp + '.csv'
    elif func.__name__ == 'group_sequential':
        results = Parallel(n_jobs=int(cpus))(
            delayed(func)(s,
                          d,
                          kwargs['kpi'])
            for s in range(sims) for d in range(days)
        )
        filename = func.__name__ + '/' + func.__name__ + '_' + kwargs['kpi'] + '_' + timestamp + '.csv'
    else:
        raise NotImplementedError

    end_time = datetime.datetime.now()
    time_used = end_time - start_time
    print("All time spent in seconds:" + str(time_used.seconds))

    filename = 'output/' + filename
    relativeBasedir = os.path.dirname(filename)
    if not os.path.exists(relativeBasedir):
        os.makedirs(relativeBasedir)

    with open(filename, 'w+') as file_handler:
        out = csv.writer(file_handler)
        for item in results:
            out.writerow(item)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-c", "--cpu", dest="cpu",
                      help="Number of CPUs to use, default=1", default=1, type='int')
    parser.add_option("-f", "--func", dest="func",
                      help="Function that implements the stopping criteria", type='str')
    parser.add_option("-k", "--kpi", dest="kpi",
                      help="Name of the KPI to be evaluated, default=normal_same", default='normal_same', type='str')
    parser.add_option("-m", "--model", dest="model_file",
                      help="Precompiled Stan model file name", default='', type='str')
    parser.add_option("--distribution", dest="distribution",
                      help="Type of the prior distribution", default='cauchy', type='str')
    parser.add_option("--scale", dest="scale",
                      help="Scale parameter of the prior distribution", default=1, type='float')
    (options, args) = parser.parse_args()

    func = eval(options.func)
    run(func,
        options.cpu,
        model_file=options.model_file,
        kpi=options.kpi,)
