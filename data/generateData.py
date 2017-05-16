import os
import numpy as np
import pandas as pd
from scipy.stats import poisson, uniform, norm, binom

sims = 1000
days = 20
total_entities = 20000  # entity means e.g. person
averageVisitPerEntity = 3

Z_alpha = 1.64485
Z_beta = -0.8416
r = 0.6 / 0.4
N = averageVisitPerEntity * total_entities  # expectation

Pc = 0.5
Pt = 0.7
Vc = 10
Vt = 20

deltaGaussian = 0.04     # must be larger than effectSizeGuassian()
deltaBinomial = Pt - Pc  # must be larger than effectSizeBinomial()

"""
Generate zero-truncated Poisson random numbers.
"""
def randomNumberPoisson(lam):
    lower = poisson.pmf(0, lam)
    return poisson.ppf(uniform.rvs(size=1, loc=lower, scale=1 - lower), lam)


"""
Compute effect size for simulated gaussian distributed kpi.
"""
def effectSizeGuassian():
    variance = 1
    n_c = N / (1+r)   # expectation
    return powerAnalysis(Z_alpha, Z_beta, variance, variance, n_c, r)


"""
Compute effect size for simulated binomial distributed kpi.
"""
def effectSizeBinomial():
    variance_c = (1/Vc)**2 * (Vc * Pc * (1-Pc))
    variance_t = (1/Vt)**2 * (Vt * Pt * (1-Pt))
    n_c = N / (1+r)   # expectation
    return powerAnalysis(Z_alpha, Z_beta, variance_c, variance_t, n_c, r)


"""
Power analysis: 
    given z_alpha, z_beta, sample variance, control sample size, ratio of treatment with control
    return effect size
"""
def powerAnalysis(Z_alpha, Z_beta, variance_t, variance_c, n_c, r):
    delta2 = ((Z_alpha-Z_beta) ** 2 * (variance_c + variance_t/r)) / n_c
    return np.sqrt(delta2)


def saveAsFile(filePath, data):
    relativeBasedir = os.path.dirname(filePath)
    if not os.path.exists(relativeBasedir):
        os.makedirs(relativeBasedir)
    data.to_csv(filePath)


def generate_random_data(seed):
    np.random.seed(seed)
    assignment = pd.DataFrame({'entity': range(total_entities),
                               'variant': np.random.choice(['A', 'B'], size=total_entities, p=[0.6, 0.4])})

    all_data = pd.DataFrame()
    for e in range(total_entities):
        n_for_e = int(randomNumberPoisson(averageVisitPerEntity))
        if n_for_e > days:
            n_for_e = days

        # an entity/person visit in these days: timePoints
        timePoints = np.random.choice(days, size=n_for_e, replace=False)

        normal_same_rv = norm.rvs(size=n_for_e, loc=0)
        normal_shifted_rv = norm.rvs(size=n_for_e, loc=0)
        conversion_rate_same_rv = binom.rvs(Vc, Pc, size=n_for_e) / Vc
        conversion_rate_shifted_rv = binom.rvs(Vc, Pc, size=n_for_e) / Vc

        # treatment variant
        if assignment.variant[assignment.entity == e].iloc[0] == 'B':
            normal_shifted_rv = norm.rvs(size=n_for_e, loc=deltaGaussian)
            conversion_rate_shifted_rv = binom.rvs(Vt, Pt, size=n_for_e) / Vt

        df = pd.DataFrame({
            'entity': e,
            'normal_same': normal_same_rv,
            'normal_shifted': normal_shifted_rv,
            'cr_same': conversion_rate_same_rv,
            'cr_shifted': conversion_rate_shifted_rv,
            'time': timePoints
        })
        all_data = all_data.append(df, ignore_index=True)

    all_data = pd.merge(all_data, assignment, on='entity')
    return all_data


if __name__ == "__main__":
    effectSizeGaussian = effectSizeGuassian()
    print("min effect size (difference of mean value) of simulated Gaussian kpi:", effectSizeGaussian)

    effectSizeBinomial = effectSizeBinomial()
    print("min effect size (difference of conversion rate) of simulated binomial kpi:", effectSizeBinomial)

    # for sim in range(sims):
    #     data = generate_random_data(sim)
    #     saveAsFile("simulation/simulation"+str(sim)+".csv", data)
