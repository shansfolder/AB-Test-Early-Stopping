import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readData(fileName):
    return pd.read_csv(fileName)


def evaluateEarlyStopping(data_negative, data_positive, idx_column_day, idx_column_significance):
    return




if __name__ == "__main__":
    normal_same_frequentist = "../output/group_sequential/group_sequential_normal_same.csv"
    normal_shifted_frequentist = "../output/group_sequential/group_sequential_normal_shifted.csv"
    normal_same_bayesian = "../output/bayes_factor/bayes_factor_normal_same.csv"
    normal_shifted_bayesian = "../output/bayes_factor/bayes_factor_normal_shifted.csv"

    data_negative_frequentist = readData(normal_same_frequentist)
    data_positive_frequentist = readData(normal_shifted_frequentist)
    data_negative_bayesian = readData(normal_same_bayesian)
    data_positive_bayesian = readData(normal_shifted_bayesian)