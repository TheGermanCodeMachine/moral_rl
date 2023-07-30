import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent.parent
sys.path.append(str(adjacent_folder))

import pickle
import numpy as np
from helpers.folder_util_functions import read, write
from tabulate import tabulate

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

analyse_data_mixtures()