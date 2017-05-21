import pandas as pd


def readData(fileName):
    return pd.read_csv(fileName)


def characteristics(data_negative, data_positive, idx_column_day, idx_column_significance):
    # 1000 negative data
    data_neg = data_negative[data_negative[idx_column_day]==19]
    n_tn = data_neg[data_neg[idx_column_significance] == False].shape[0]
    n_fp = data_neg[data_neg[idx_column_significance] == True].shape[0]

    # 1000 positive data
    data_pos = data_positive[data_positive[idx_column_day]==19]
    n_fn = data_pos[data_pos[idx_column_significance] == False].shape[0]
    n_tp = data_pos[data_pos[idx_column_significance] == True].shape[0]
    return n_fp, n_tn, n_fn, n_tp

if __name__ == "__main__":
    normal_same_frequentist = "../output/group_sequential/group_sequential_normal_same.csv"
    normal_shifted_frequentist = "../output/group_sequential/group_sequential_normal_shifted.csv"
    normal_same_bayesian = "../output/bayes_factor/bayes_factor_normal_same.csv"
    normal_shifted_bayesian = "../output/bayes_factor/bayes_factor_normal_shifted.csv"

    data_negative_frequentist = readData(normal_same_frequentist)
    data_positive_frequentist = readData(normal_shifted_frequentist)
    data_negative_bayesian = readData(normal_same_bayesian)
    data_positive_bayesian = readData(normal_shifted_bayesian)

    n_fp, n_tn, n_fn, n_tp = characteristics(data_negative_frequentist, data_positive_frequentist, 'day_idx', 'significant')
    print("group sequential: ", n_fp, n_tn, n_fn, n_tp)

    n_fp, n_tn, n_fn, n_tp = characteristics(data_negative_bayesian, data_positive_bayesian, 'day_idx', 'significant_ci')
    print("credible interval: ", n_fp, n_tn, n_fn, n_tp)

    n_fp, n_tn, n_fn, n_tp = characteristics(data_negative_bayesian, data_positive_bayesian, 'day_idx', 'significant_and_stop_bf')
    print("bayes factor: ", n_fp, n_tn, n_fn, n_tp)
