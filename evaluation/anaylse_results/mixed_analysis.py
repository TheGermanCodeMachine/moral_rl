import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent.parent
sys.path.append(str(adjacent_folder))

import pickle
import numpy as np
from helpers.folder_util_functions import read, write
from tabulate import tabulate
import matplotlib.pyplot as plt

def analyse_data_mixtures():
    base_path = "datasets\\100_ablations_3\\pvcd100\\results_mixedNN\\"
    data_sin_01 = read(base_path + "0,1\single__counterfactual.pkl")
    data_sin_052 = read(base_path + "0.5,1\single__counterfactual.pkl")
    data_sin_0251 = read(base_path + "0.25,1\single__counterfactual.pkl")
    data_sin_0751 = read(base_path + "0.75,1\single__counterfactual.pkl")
    data_sin_10 = read(base_path + "1,0\single__counterfactual.pkl")
    data_sin_105 = read(base_path + "1,0.5\single__counterfactual.pkl")
    data_sin_1025 = read(base_path + "1,0.25\single__counterfactual.pkl")
    data_sin_1075 = read(base_path + "1,0.75\single__counterfactual.pkl")
    data_sin_11 = read(base_path + "1,1\single__counterfactual.pkl")


    data_con_01 = read(base_path + "0,1\contrastive__counterfactual.pkl")
    data_con_052 = read(base_path + "0.5,1\contrastive__counterfactual.pkl")
    data_con_0251 = read(base_path + "0.25,1\contrastive__counterfactual.pkl")
    data_con_0751 = read(base_path + "0.75,1\contrastive__counterfactual.pkl")
    data_con_10 = read(base_path + "1,0\contrastive__counterfactual.pkl")
    data_con_105 = read(base_path + "1,0.5\contrastive__counterfactual.pkl")
    data_con_1025 = read(base_path + "1,0.25\contrastive__counterfactual.pkl")
    data_con_1075 = read(base_path + "1,0.75\contrastive__counterfactual.pkl")
    data_con_11 = read(base_path + "1,1\contrastive__counterfactual.pkl")

    data_sin_01_ood = read(base_path + "0,1\single__counterfactual_ood.pkl")
    data_sin_052_ood = read(base_path + "0.5,1\single__counterfactual_ood.pkl")
    data_sin_0251_ood = read(base_path + "0.25,1\single__counterfactual_ood.pkl")
    data_sin_0751_ood = read(base_path + "0.75,1\single__counterfactual_ood.pkl")
    data_sin_10_ood = read(base_path + "1,0\single__counterfactual_ood.pkl")
    data_sin_105_ood = read(base_path + "1,0.5\single__counterfactual_ood.pkl")
    data_sin_1025_ood = read(base_path + "1,0.25\single__counterfactual_ood.pkl")
    data_sin_1075_ood = read(base_path + "1,0.75\single__counterfactual_ood.pkl")
    data_sin_11_ood = read(base_path + "1,1\single__counterfactual_ood.pkl")


    data_con_01_ood = read(base_path + "0,1\contrastive__counterfactual_ood.pkl")
    data_con_052_ood = read(base_path + "0.5,1\contrastive__counterfactual_ood.pkl")
    data_con_0251_ood = read(base_path + "0.25,1\contrastive__counterfactual_ood.pkl")
    data_con_0751_ood = read(base_path + "0.75,1\contrastive__counterfactual_ood.pkl")
    data_con_10_ood = read(base_path + "1,0\contrastive__counterfactual_ood.pkl")
    data_con_105_ood = read(base_path + "1,0.5\contrastive__counterfactual_ood.pkl")
    data_con_1025_ood = read(base_path + "1,0.25\contrastive__counterfactual_ood.pkl")
    data_con_1075_ood = read(base_path + "1,0.75\contrastive__counterfactual_ood.pkl")
    data_con_11_ood = read(base_path + "1,1\contrastive__counterfactual_ood.pkl")

    # print("0,1", round(np.mean(data_sin_01['test_mean_errors']),2), round(np.mean(data_con_01['test_mean_errors']),2), round(np.mean(data_sin_01_ood['test_mean_errors']),2), round(np.mean(data_con_01_ood['test_mean_errors'])))
    # print("0.25,1", round(np.mean(data_sin_0251['test_mean_errors']),2), round(np.mean(data_con_0251['test_mean_errors']),2), round(np.mean(data_sin_0251_ood['test_mean_errors']),2), round(np.mean(data_con_0251_ood['test_mean_errors'])))
    # print("0.5,1", round(np.mean(data_sin_052['test_mean_errors']),2), round(np.mean(data_con_052['test_mean_errors']),2), round(np.mean(data_sin_052_ood['test_mean_errors']),2), round(np.mean(data_con_052_ood['test_mean_errors'])))
    # print("0.75,1", round(np.mean(data_sin_0751['test_mean_errors']),2), round(np.mean(data_con_0751['test_mean_errors']),2), round(np.mean(data_sin_0751_ood['test_mean_errors']),2), round(np.mean(data_con_0751_ood['test_mean_errors'])))
    # print("1,1", round(np.mean(data_sin_11['test_mean_errors']),2), round(np.mean(data_con_11['test_mean_errors']),2), round(np.mean(data_sin_11_ood['test_mean_errors']),2), round(np.mean(data_con_11_ood['test_mean_errors'])))
    # print("1,0.75", round(np.mean(data_sin_1075['test_mean_errors']),2), round(np.mean(data_con_1075['test_mean_errors']),2), round(np.mean(data_sin_1075_ood['test_mean_errors']),2), round(np.mean(data_con_1075_ood['test_mean_errors'])))
    # print("1,0.5", round(np.mean(data_sin_105['test_mean_errors']),2), round(np.mean(data_con_105['test_mean_errors']),2), round(np.mean(data_sin_105_ood['test_mean_errors']),2), round(np.mean(data_con_105_ood['test_mean_errors'])))
    # print("1,0.25", round(np.mean(data_sin_1025['test_mean_errors']),2), round(np.mean(data_con_1025['test_mean_errors']),2), round(np.mean(data_sin_1025_ood['test_mean_errors']),2), round(np.mean(data_con_1025_ood['test_mean_errors'])))
    # print("1,0", round(np.mean(data_sin_10['test_mean_errors']),2), round(np.mean(data_con_10['test_mean_errors']),2), round(np.mean(data_sin_10_ood['test_mean_errors']),2), round(np.mean(data_con_10_ood['test_mean_errors'])))
    # # print the results so they are aligned in the output

    table = [
        ["mix (c,s)", "contrastive", "single", "contrastive ood", "single ood"],
        ["0,1", round(np.mean(data_con_01['test_mean_errors']),2), round(np.mean(data_sin_01['test_mean_errors']),2), round(np.mean(data_con_01_ood['test_mean_errors']),2), round(np.mean(data_sin_01_ood['test_mean_errors']),2)],
        ["0.25,1", round(np.mean(data_con_0251['test_mean_errors']),2), round(np.mean(data_sin_0251['test_mean_errors']),2), round(np.mean(data_con_0251_ood['test_mean_errors']),2), round(np.mean(data_sin_0251_ood['test_mean_errors']),2)],
        ["0.5,1", round(np.mean(data_con_052['test_mean_errors']),2), round(np.mean(data_sin_052['test_mean_errors']),2), round(np.mean(data_con_052_ood['test_mean_errors']),2), round(np.mean(data_sin_052_ood['test_mean_errors']),2)],
        ["0.75,1", round(np.mean(data_con_0751['test_mean_errors']),2), round(np.mean(data_sin_0751['test_mean_errors']),2), round(np.mean(data_con_0751_ood['test_mean_errors']),2), round(np.mean(data_sin_0751_ood['test_mean_errors']),2)],
        ["1,1", round(np.mean(data_con_11['test_mean_errors']),2), round(np.mean(data_sin_11['test_mean_errors']),2), round(np.mean(data_con_11_ood['test_mean_errors']),2), round(np.mean(data_sin_11_ood['test_mean_errors']),2)],
        ["1,0.75", round(np.mean(data_con_1075['test_mean_errors']),2), round(np.mean(data_sin_1075['test_mean_errors']),2), round(np.mean(data_con_1075_ood['test_mean_errors']),2), round(np.mean(data_sin_1075_ood['test_mean_errors']),2)],
        ["1,0.5", round(np.mean(data_con_105['test_mean_errors']),2), round(np.mean(data_sin_105['test_mean_errors']),2), round(np.mean(data_con_105_ood['test_mean_errors']),2), round(np.mean(data_sin_105_ood['test_mean_errors']),2)],
        ["1,0.25", round(np.mean(data_con_1025['test_mean_errors']),2), round(np.mean(data_sin_1025['test_mean_errors']),2), round(np.mean(data_con_1025_ood['test_mean_errors']),2), round(np.mean(data_sin_1025_ood['test_mean_errors']),2)],
        ["1,0", round(np.mean(data_con_10['test_mean_errors']),2), round(np.mean(data_sin_10['test_mean_errors']),2), round(np.mean(data_con_10_ood['test_mean_errors']),2), round(np.mean(data_sin_10_ood['test_mean_errors']),2)],
    ]
    print(tabulate(table))

def analyse_task_weights():
    base_path = "datasets\\100_ablations_3\\pvcd100\\results_mixedNN\\(1,1)task_weights\\"
    data_sin_1 = read(base_path + "1,1\single__counterfactual.pkl")
    data_sin_3 = read(base_path + "1,3\single__counterfactual.pkl")
    data_sin_9 = read(base_path + "1,9\single__counterfactual.pkl")
    data_sin_15 = read(base_path + "1,15\single__counterfactual.pkl")
    data_sin_28 = read(base_path + "1,28\single__counterfactual.pkl")
    data_sin_40 = read(base_path + "1,40\single__counterfactual.pkl")
    data_sin_60 = read(base_path + "1,60\single__counterfactual.pkl")

    data_con_1 = read(base_path + "1,1\contrastive__counterfactual.pkl")
    data_con_3 = read(base_path + "1,3\contrastive__counterfactual.pkl")
    data_con_9 = read(base_path + "1,9\contrastive__counterfactual.pkl")
    data_con_15 = read(base_path + "1,15\contrastive__counterfactual.pkl")
    data_con_28 = read(base_path + "1,28\contrastive__counterfactual.pkl")
    data_con_40 = read(base_path + "1,40\contrastive__counterfactual.pkl")
    data_con_60 = read(base_path + "1,60\contrastive__counterfactual.pkl")
    

    data_sin_1_ood = read(base_path + "1,1\single__counterfactual_ood.pkl")
    data_sin_3_ood = read(base_path + "1,3\single__counterfactual_ood.pkl")
    data_sin_9_ood = read(base_path + "1,9\single__counterfactual_ood.pkl")
    data_sin_15_ood = read(base_path + "1,15\single__counterfactual_ood.pkl")
    data_sin_28_ood = read(base_path + "1,28\single__counterfactual_ood.pkl")
    data_sin_40_ood = read(base_path + "1,40\single__counterfactual_ood.pkl")
    data_sin_60_ood = read(base_path + "1,60\single__counterfactual_ood.pkl")

    data_con_1_ood = read(base_path + "1,1\contrastive__counterfactual_ood.pkl")
    data_con_3_ood = read(base_path + "1,3\contrastive__counterfactual_ood.pkl")
    data_con_9_ood = read(base_path + "1,9\contrastive__counterfactual_ood.pkl")
    data_con_15_ood = read(base_path + "1,15\contrastive__counterfactual_ood.pkl")
    data_con_28_ood = read(base_path + "1,28\contrastive__counterfactual_ood.pkl")
    data_con_40_ood = read(base_path + "1,40\contrastive__counterfactual_ood.pkl")
    data_con_60_ood = read(base_path + "1,60\contrastive__counterfactual_ood.pkl")

    table = [
        ["weights (c,s)", "contrastive", "single", "contrastive ood", "single ood"],
        ["1,1", round(np.mean(data_con_1['test_mean_errors']),2), round(np.mean(data_sin_1['test_mean_errors']),2), round(np.mean(data_con_1_ood['test_mean_errors']),2), round(np.mean(data_sin_1_ood['test_mean_errors']),2)],
        ["1,3", round(np.mean(data_con_3['test_mean_errors']),2), round(np.mean(data_sin_3['test_mean_errors']),2), round(np.mean(data_con_3_ood['test_mean_errors']),2), round(np.mean(data_sin_3_ood['test_mean_errors']),2)],
        ["1,9", round(np.mean(data_con_9['test_mean_errors']),2), round(np.mean(data_sin_9['test_mean_errors']),2), round(np.mean(data_con_9_ood['test_mean_errors']),2), round(np.mean(data_sin_9_ood['test_mean_errors']),2)],
        ["1,15", round(np.mean(data_con_15['test_mean_errors']),2), round(np.mean(data_sin_15['test_mean_errors']),2), round(np.mean(data_con_15_ood['test_mean_errors']),2), round(np.mean(data_sin_15_ood['test_mean_errors']),2)],
        ["1,28", round(np.mean(data_con_28['test_mean_errors']),2), round(np.mean(data_sin_28['test_mean_errors']),2), round(np.mean(data_con_28_ood['test_mean_errors']),2), round(np.mean(data_sin_28_ood['test_mean_errors']),2)],
        ["1,40", round(np.mean(data_con_40['test_mean_errors']),2), round(np.mean(data_sin_40['test_mean_errors']),2), round(np.mean(data_con_40_ood['test_mean_errors']),2), round(np.mean(data_sin_40_ood['test_mean_errors']),2)],
        ["1,60", round(np.mean(data_con_60['test_mean_errors']),2), round(np.mean(data_sin_60['test_mean_errors']),2), round(np.mean(data_con_60_ood['test_mean_errors']),2), round(np.mean(data_sin_60_ood['test_mean_errors']),2)],
    ]

    print(tabulate(table))

def analyse_num_data():
    base_path = "datasets\\1000\\1000\\results_sidebyside"
    it = [5,10,20,50,100,200,500,800]
    data_5 = read(base_path + "\\5\\contrastive__counterfactual.pkl")
    data_5_ood = read(base_path + "\\5\\contrastive__counterfactual_ood.pkl")
    data_10 = read(base_path + "\\10\\contrastive__counterfactual.pkl")
    data_10_ood = read(base_path + "\\10\\contrastive__counterfactual_ood.pkl")
    data_20 = read(base_path + "\\20\\contrastive__counterfactual.pkl")
    data_20_ood = read(base_path + "\\20\\contrastive__counterfactual_ood.pkl")
    data_50 = read(base_path + "\\50\\contrastive__counterfactual.pkl")
    data_50_ood = read(base_path + "\\50\\contrastive__counterfactual_ood.pkl")
    data_100 = read(base_path + "\\100\\contrastive__counterfactual.pkl")
    data_100_ood = read(base_path + "\\100\\contrastive__counterfactual_ood.pkl")
    data_200 = read(base_path + "\\200\\contrastive__counterfactual.pkl")
    data_200_ood = read(base_path + "\\200\\contrastive__counterfactual_ood.pkl")
    data_500 = read(base_path + "\\500\\contrastive__counterfactual.pkl")
    data_500_ood = read(base_path + "\\500\\contrastive__counterfactual_ood.pkl")
    data_800 = read(base_path + "\\800\\contrastive__counterfactual.pkl")
    data_800_ood = read(base_path + "\\800\\contrastive__counterfactual_ood.pkl")

    base_path = "datasets\\1000\\1000\\results_sidebysideLM"
    it = [5,10,20,50,100,200,500,800]
    data_LM_5 = read(base_path + "\\5\\contrastive__counterfactual.pkl")
    data_LM_5_ood = read(base_path + "\\5\\contrastive__counterfactual_ood.pkl")
    data_LM_10 = read(base_path + "\\10\\contrastive__counterfactual.pkl")
    data_LM_10_ood = read(base_path + "\\10\\contrastive__counterfactual_ood.pkl")
    data_LM_20 = read(base_path + "\\20\\contrastive__counterfactual.pkl")
    data_LM_20_ood = read(base_path + "\\20\\contrastive__counterfactual_ood.pkl")
    data_LM_50 = read(base_path + "\\50\\contrastive__counterfactual.pkl")
    data_LM_50_ood = read(base_path + "\\50\\contrastive__counterfactual_ood.pkl")
    data_LM_100 = read(base_path + "\\100\\contrastive__counterfactual.pkl")
    data_LM_100_ood = read(base_path + "\\100\\contrastive__counterfactual_ood.pkl")
    data_LM_200 = read(base_path + "\\200\\contrastive__counterfactual.pkl")
    data_LM_200_ood = read(base_path + "\\200\\contrastive__counterfactual_ood.pkl")
    data_LM_500 = read(base_path + "\\500\\contrastive__counterfactual.pkl")
    data_LM_500_ood = read(base_path + "\\500\\contrastive__counterfactual_ood.pkl")
    data_LM_800 = read(base_path + "\\800\\contrastive__counterfactual.pkl")
    data_LM_800_ood = read(base_path + "\\800\\contrastive__counterfactual_ood.pkl")

    table = [
        ["amount of data", "contrastive", "contrastive ood"],
        ["5", round(np.mean(data_5['test_mean_errors']),2), round(np.mean(data_5_ood['test_mean_errors']),2)],
        ["10", round(np.mean(data_10['test_mean_errors']),2), round(np.mean(data_10_ood['test_mean_errors']),2)],
        ["20", round(np.mean(data_20['test_mean_errors']),2), round(np.mean(data_20_ood['test_mean_errors']),2)],
        ["50", round(np.mean(data_50['test_mean_errors']),2), round(np.mean(data_50_ood['test_mean_errors']),2)],
        ["100", round(np.mean(data_100['test_mean_errors']),2), round(np.mean(data_100_ood['test_mean_errors']),2)],
        ["200", round(np.mean(data_200['test_mean_errors']),2), round(np.mean(data_200_ood['test_mean_errors']),2)],
        ["500", round(np.mean(data_500['test_mean_errors']),2), round(np.mean(data_500_ood['test_mean_errors']),2)],
        ["800", round(np.mean(data_800['test_mean_errors']),2), round(np.mean(data_800_ood['test_mean_errors']),2)],
    ]
    print(tabulate(table))

    table = [
        ["amount of data", "contrastive", "contrastive ood"],
        ["5", round(np.mean(data_5['spearman_correlations']),2), round(np.mean(data_5_ood['spearman_correlations']),2)],
        ["10", round(np.mean(data_10['spearman_correlations']),2), round(np.mean(data_10_ood['spearman_correlations']),2)],
        ["20", round(np.mean(data_20['spearman_correlations']),2), round(np.mean(data_20_ood['spearman_correlations']),2)],
        ["50", round(np.mean(data_50['spearman_correlations']),2), round(np.mean(data_50_ood['spearman_correlations']),2)],
        ["100", round(np.mean(data_100['spearman_correlations']),2), round(np.mean(data_100_ood['spearman_correlations']),2)],
        ["200", round(np.mean(data_200['spearman_correlations']),2), round(np.mean(data_200_ood['spearman_correlations']),2)],
        ["500", round(np.mean(data_500['spearman_correlations']),2), round(np.mean(data_500_ood['spearman_correlations']),2)],
        ["800", round(np.mean(data_800['spearman_correlations']),2), round(np.mean(data_800_ood['spearman_correlations']),2)],
    ]
    print(tabulate(table))

    # plot the values from the table
    # test_mean_errors = [np.mean(data_5["test_mean_errors"]), np.mean(data_10["test_mean_errors"]), np.mean(data_20["test_mean_errors"]), np.mean(data_50["test_mean_errors"]), np.mean(data_100["test_mean_errors"]), np.mean(data_200["test_mean_errors"]), np.mean(data_500["test_mean_errors"]), np.mean(data_800["test_mean_errors"])]
    # plt.plot(it, test_mean_errors, label="test mean errors")
    # spearman_correlations = [np.mean(data_5["spearman_correlations"]), np.mean(data_10["spearman_correlations"]), np.mean(data_20["spearman_correlations"]), np.mean(data_50["spearman_correlations"]), np.mean(data_100["spearman_correlations"]), np.mean(data_200["spearman_correlations"]), np.mean(data_500["spearman_correlations"]), np.mean(data_800["spearman_correlations"])]
    # plt.plot(it, spearman_correlations, label="spearman correlations")
    # pearson_correlations = [np.mean(data_5["pearson_correlations"]), np.mean(data_10["pearson_correlations"]), np.mean(data_20["pearson_correlations"]), np.mean(data_50["pearson_correlations"]), np.mean(data_100["pearson_correlations"]), np.mean(data_200["pearson_correlations"]), np.mean(data_500["pearson_correlations"]), np.mean(data_800["pearson_correlations"])]
    # plt.plot(it, pearson_correlations, label="pearson correlations")
    # r2s = [np.mean(data_5["r2s"]), np.mean(data_10["r2s"]), 0, np.mean(data_50["r2s"]), np.mean(data_100["r2s"]), np.mean(data_200["r2s"]), np.mean(data_500["r2s"]), np.mean(data_800["r2s"])]
    # plt.plot(it, r2s, label="r2s")
    # plt.legend()
    # plt.show()

    # test_mean_error_ood = [np.mean(data_5_ood["test_mean_errors"]), np.mean(data_10_ood["test_mean_errors"]), np.mean(data_20_ood["test_mean_errors"]), np.mean(data_50_ood["test_mean_errors"]), np.mean(data_100_ood["test_mean_errors"]), np.mean(data_200_ood["test_mean_errors"]), np.mean(data_500_ood["test_mean_errors"]), np.mean(data_800_ood["test_mean_errors"])]
    # plt.plot(it, test_mean_error_ood, label="test mean errors")
    # spearman_correlations_ood = [np.mean(data_5_ood["spearman_correlations"]), np.mean(data_10_ood["spearman_correlations"]), np.mean(data_20_ood["spearman_correlations"]), np.mean(data_50_ood["spearman_correlations"]), np.mean(data_100_ood["spearman_correlations"]), np.mean(data_200_ood["spearman_correlations"]), np.mean(data_500_ood["spearman_correlations"]), np.mean(data_800_ood["spearman_correlations"])]
    # plt.plot(it, spearman_correlations_ood, label="spearman correlations")
    # pearson_correlations_ood = [np.mean(data_5_ood["pearson_correlations"]), np.mean(data_10_ood["pearson_correlations"]), np.mean(data_20_ood["pearson_correlations"]), np.mean(data_50_ood["pearson_correlations"]), np.mean(data_100_ood["pearson_correlations"]), np.mean(data_200_ood["pearson_correlations"]), np.mean(data_500_ood["pearson_correlations"]), np.mean(data_800_ood["pearson_correlations"])]
    # plt.plot(it, pearson_correlations_ood, label="pearson correlations")
    # r2s_ood = [np.mean(data_5_ood["r2s"]), np.mean(data_10_ood["r2s"]), 0, np.mean(data_50_ood["r2s"]), np.mean(data_100_ood["r2s"]), np.mean(data_200_ood["r2s"]), np.mean(data_500_ood["r2s"]), np.mean(data_800_ood["r2s"])]
    # plt.plot(it, r2s_ood, label="r2s")
    # plt.legend()
    # plt.show()

    test_mean_errors = [np.mean(data_5["test_mean_errors"]), np.mean(data_10["test_mean_errors"]), np.mean(data_20["test_mean_errors"]), np.mean(data_50["test_mean_errors"]), np.mean(data_100["test_mean_errors"]), np.mean(data_200["test_mean_errors"]), np.mean(data_500["test_mean_errors"]), np.mean(data_800["test_mean_errors"])]
    test_mean_errorsLM = [np.mean(data_LM_5["test_mean_errors"]), np.mean(data_LM_10["test_mean_errors"]), np.mean(data_LM_20["test_mean_errors"]), np.mean(data_LM_50["test_mean_errors"]), np.mean(data_LM_100["test_mean_errors"]), np.mean(data_LM_200["test_mean_errors"]), np.mean(data_LM_500["test_mean_errors"]), np.mean(data_LM_800["test_mean_errors"])]
    plt.plot(it, test_mean_errors, label="NN")
    plt.plot(it, test_mean_errorsLM, label="LM")
    plt.legend()
    plt.title("Test mean errors for different amounts of training data")
    plt.xlabel("Amount of training data")
    plt.ylabel("Test mean errors")
    plt.show()

    test_mean_errors_ood = [np.mean(data_5_ood["test_mean_errors"]), np.mean(data_10_ood["test_mean_errors"]), np.mean(data_20_ood["test_mean_errors"]), np.mean(data_50_ood["test_mean_errors"]), np.mean(data_100_ood["test_mean_errors"]), np.mean(data_200_ood["test_mean_errors"]), np.mean(data_500_ood["test_mean_errors"]), np.mean(data_800_ood["test_mean_errors"])]
    test_mean_errorsLM_ood = [np.mean(data_LM_5_ood["test_mean_errors"]), np.mean(data_LM_10_ood["test_mean_errors"]), np.mean(data_LM_20_ood["test_mean_errors"]), np.mean(data_LM_50_ood["test_mean_errors"]), np.mean(data_LM_100_ood["test_mean_errors"]), np.mean(data_LM_200_ood["test_mean_errors"]), np.mean(data_LM_500_ood["test_mean_errors"]), np.mean(data_LM_800_ood["test_mean_errors"])]
    plt.plot(it, test_mean_errors_ood, label="NN")
    plt.plot(it, test_mean_errorsLM_ood, label="LM")
    plt.legend()
    plt.title("OOD Test mean errors for different amounts of training data")
    plt.xlabel("Amount of training data")
    plt.ylabel("Test mean errors")
    plt.show()


analyse_task_weights()