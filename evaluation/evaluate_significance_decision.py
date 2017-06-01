import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def readData(fileName):
    return pd.read_csv(fileName)


def characteristics(data_negative, data_positive, idx_column_day, idx_column_significance, endDate=19):
    # 1000 negative data
    data_neg = data_negative[data_negative[idx_column_day] == endDate]
    n_tn = data_neg[data_neg[idx_column_significance] == False].shape[0]
    n_fp = data_neg[data_neg[idx_column_significance] == True].shape[0]

    # 1000 positive data
    data_pos = data_positive[data_positive[idx_column_day] == endDate]
    n_fn = data_pos[data_pos[idx_column_significance] == False].shape[0]
    n_tp = data_pos[data_pos[idx_column_significance] == True].shape[0]
    return n_fp, n_tn, n_fn, n_tp


def plotInRoc(freq_fpr, freq_tpr, credIn_fpr, credIn_tpr, bf_fpr, bf_tpr):
    plt.plot(freq_fpr, freq_tpr, 'mx', label="confidence interval", markersize=8)
    plt.plot(credIn_fpr, credIn_tpr, 'b+', label="credible interval", markersize=10)
    plt.plot(bf_fpr, bf_tpr, 'r*', label="Bayes factor", markersize=10)
    x = np.linspace(0.0, 1.0, 100)
    plt.plot(x, x, 'c--', linewidth=0.5, label="random guess")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title('ROC plot of different significance decision')
    plt.legend(loc=4)
    plt.show()
    return


def evaluateSimulationData():
    normal_same_frequentist = "../output/group_sequential/group_sequential_normal_same.csv"
    normal_shifted_frequentist = "../output/group_sequential/group_sequential_normal_shifted.csv"
    normal_same_bayesian = "../output/bayes_factor/bayes_factor_normal_same.csv"
    normal_shifted_bayesian = "../output/bayes_factor/bayes_factor_normal_shifted.csv"

    data_negative_frequentist = readData(normal_same_frequentist)
    data_positive_frequentist = readData(normal_shifted_frequentist)
    data_negative_bayesian = readData(normal_same_bayesian)
    data_positive_bayesian = readData(normal_shifted_bayesian)

    n_fp, n_tn, n_fn, n_tp = characteristics(data_negative_frequentist, data_positive_frequentist, 'day_idx',
                                             'significant')
    print("group sequential: ", n_fp, n_tn, n_fn, n_tp)

    freq_fpr = n_fp / 1000.
    freq_tpr = n_tp / 1000.

    n_fp, n_tn, n_fn, n_tp = characteristics(data_negative_bayesian, data_positive_bayesian, 'day_idx',
                                             'significant_ci')
    print("credible interval: ", n_fp, n_tn, n_fn, n_tp)

    credIn_fpr = n_fp / 1000.
    credIn_tpr = n_tp / 1000.

    n_fp, n_tn, n_fn, n_tp = characteristics(data_negative_bayesian, data_positive_bayesian, 'day_idx',
                                             'significant_and_stop_bf')
    print("bayes factor: ", n_fp, n_tn, n_fn, n_tp)

    bf_fpr = n_fp / 1000.
    bf_tpr = n_tp / 1000.

    plotInRoc(freq_fpr, freq_tpr, credIn_fpr, credIn_tpr, bf_fpr, bf_tpr)


if __name__ == "__main__":
    evaluateSimulationData()

