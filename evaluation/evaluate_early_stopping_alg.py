import pandas as pd

def readData(fileName):
    return pd.read_csv(fileName)


def computeEvaluationMetrics(data, idx_column_stop, idx_column_significance):
    falseStoppedCount = 0
    stoppedCount = 0
    daysCount_tp = 0
    daysCount = 0
    allDays = 20*1000
    bias_sum = 0

    for test in range(1000):
        cur_test = data[data.test_idx == test]
        last_day = cur_test[cur_test.day_idx == 19]
        last_day_significant = last_day.iloc[0][idx_column_significance]
        last_day_effect_size = last_day.iloc[0]["mean_delta"]
        early_stopped = cur_test[cur_test[idx_column_stop] == True].shape[0] > 0

        if early_stopped:
            stoppedCount += 1
            stop_day = cur_test[cur_test[idx_column_stop] == True].min()["day_idx"]

            significant_stop = cur_test[cur_test.day_idx == stop_day].iloc[0][idx_column_significance]
            if significant_stop != last_day_significant:
                falseStoppedCount += 1
            else:
                daysCount_tp += (stop_day + 1)

            effect_size_stop = cur_test[cur_test.day_idx == stop_day].iloc[0]["mean_delta"]
            bias_sum += (last_day_effect_size - effect_size_stop)

        else:
            stop_day = 19

        daysCount += (stop_day + 1)

    FPR = falseStoppedCount / 1000
    RTR_all = (allDays - daysCount) / allDays

    days_can_stop_early = 20 * (stoppedCount - falseStoppedCount)
    RTR_tp = (days_can_stop_early - daysCount_tp) / days_can_stop_early

    bias = bias_sum / stoppedCount

    return FPR, RTR_all, RTR_tp, bias


def evaluateSimulationData():
    normal_same_frequentist = "../output/group_sequential/group_sequential_normal_same.csv"
    normal_shifted_frequentist = "../output/group_sequential/group_sequential_normal_shifted.csv"
    normal_same_bayesian = "../output/bayes_factor/bayes_factor_normal_same.csv"
    normal_shifted_bayesian = "../output/bayes_factor/bayes_factor_normal_shifted.csv"

    data_negative_frequentist = readData(normal_same_frequentist)
    data_positive_frequentist = readData(normal_shifted_frequentist)
    data_negative_bayesian = readData(normal_same_bayesian)
    data_positive_bayesian = readData(normal_shifted_bayesian)

    FPR, RTR_all, RTR_tp, bias = computeEvaluationMetrics(data_negative_frequentist, "stop", "significant")
    print("frequentist A/A", FPR, RTR_all, RTR_tp, bias)

    FPR, RTR_all, RTR_tp, bias = computeEvaluationMetrics(data_positive_frequentist, "stop", "significant")
    print("frequentist A/B", FPR, RTR_all, RTR_tp, bias)

    FPR, RTR_all, RTR_tp, bias = computeEvaluationMetrics(data_negative_bayesian, "significant_ci", "significant_ci")
    print("credible interval A/A", FPR, RTR_all, RTR_tp, bias)

    FPR, RTR_all, RTR_tp, bias = computeEvaluationMetrics(data_positive_bayesian, "significant_ci", "significant_ci")
    print("credible interval A/B", FPR, RTR_all, RTR_tp, bias)

    FPR, RTR_all, RTR_tp, bias = computeEvaluationMetrics(data_negative_bayesian, "significant_and_stop_bf", "significant_and_stop_bf")
    print("bf(stop bf) A/A", FPR, RTR_all, RTR_tp, bias)

    FPR, RTR_all, RTR_tp, bias = computeEvaluationMetrics(data_positive_bayesian, "significant_and_stop_bf", "significant_and_stop_bf")
    print("bf(stop bf) A/B", FPR, RTR_all, RTR_tp, bias)

    FPR, RTR_all, RTR_tp, bias = computeEvaluationMetrics(data_negative_bayesian, "stop_bp", "significant_and_stop_bf")
    print("bf(stop bp) A/A", FPR, RTR_all, RTR_tp, bias)

    FPR, RTR_all, RTR_tp, bias = computeEvaluationMetrics(data_positive_bayesian, "stop_bp", "significant_and_stop_bf")
    print("bf(stop bp) A/B", FPR, RTR_all, RTR_tp, bias)


if __name__ == "__main__":
    evaluateSimulationData()