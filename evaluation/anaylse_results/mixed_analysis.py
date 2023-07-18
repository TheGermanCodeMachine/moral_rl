import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent.parent
sys.path.append(str(adjacent_folder))

import pickle
import numpy as np
from helpers.folder_util_functions import read, write
from tabulate import tabulate

base_path = "datasets\\100_ablations_3\\pvcd100\\results_mixedLM\\"
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

# print("0,1", round(np.mean(data_sin_01['test_losses'])), round(np.mean(data_con_01['test_losses'])), round(np.mean(data_sin_01_ood['test_losses'])), round(np.mean(data_con_01_ood['test_losses'])))
# print("0.25,1", round(np.mean(data_sin_0251['test_losses'])), round(np.mean(data_con_0251['test_losses'])), round(np.mean(data_sin_0251_ood['test_losses'])), round(np.mean(data_con_0251_ood['test_losses'])))
# print("0.5,1", round(np.mean(data_sin_052['test_losses'])), round(np.mean(data_con_052['test_losses'])), round(np.mean(data_sin_052_ood['test_losses'])), round(np.mean(data_con_052_ood['test_losses'])))
# print("0.75,1", round(np.mean(data_sin_0751['test_losses'])), round(np.mean(data_con_0751['test_losses'])), round(np.mean(data_sin_0751_ood['test_losses'])), round(np.mean(data_con_0751_ood['test_losses'])))
# print("1,1", round(np.mean(data_sin_11['test_losses'])), round(np.mean(data_con_11['test_losses'])), round(np.mean(data_sin_11_ood['test_losses'])), round(np.mean(data_con_11_ood['test_losses'])))
# print("1,0.75", round(np.mean(data_sin_1075['test_losses'])), round(np.mean(data_con_1075['test_losses'])), round(np.mean(data_sin_1075_ood['test_losses'])), round(np.mean(data_con_1075_ood['test_losses'])))
# print("1,0.5", round(np.mean(data_sin_105['test_losses'])), round(np.mean(data_con_105['test_losses'])), round(np.mean(data_sin_105_ood['test_losses'])), round(np.mean(data_con_105_ood['test_losses'])))
# print("1,0.25", round(np.mean(data_sin_1025['test_losses'])), round(np.mean(data_con_1025['test_losses'])), round(np.mean(data_sin_1025_ood['test_losses'])), round(np.mean(data_con_1025_ood['test_losses'])))
# print("1,0", round(np.mean(data_sin_10['test_losses'])), round(np.mean(data_con_10['test_losses'])), round(np.mean(data_sin_10_ood['test_losses'])), round(np.mean(data_con_10_ood['test_losses'])))
# # print the results so they are aligned in the output

table = [
    ["mix (c,s)", "contrastive", "single", "contrastive ood", "single ood"],
    ["0,1", round(np.mean(data_con_01['test_losses'])), round(np.mean(data_sin_01['test_losses'])), round(np.mean(data_con_01_ood['test_losses'])), round(np.mean(data_sin_01_ood['test_losses']))],
    ["0.25,1", round(np.mean(data_con_0251['test_losses'])), round(np.mean(data_sin_0251['test_losses'])), round(np.mean(data_con_0251_ood['test_losses'])), round(np.mean(data_sin_0251_ood['test_losses']))],
    ["0.5,1", round(np.mean(data_con_052['test_losses'])), round(np.mean(data_sin_052['test_losses'])), round(np.mean(data_con_052_ood['test_losses'])), round(np.mean(data_sin_052_ood['test_losses']))],
    ["0.75,1", round(np.mean(data_con_0751['test_losses'])), round(np.mean(data_sin_0751['test_losses'])), round(np.mean(data_con_0751_ood['test_losses'])), round(np.mean(data_sin_0751_ood['test_losses']))],
    ["1,1", round(np.mean(data_con_11['test_losses'])), round(np.mean(data_sin_11['test_losses'])), round(np.mean(data_con_11_ood['test_losses'])), round(np.mean(data_sin_11_ood['test_losses']))],
    ["1,0.75", round(np.mean(data_con_1075['test_losses'])), round(np.mean(data_sin_1075['test_losses'])), round(np.mean(data_con_1075_ood['test_losses'])), round(np.mean(data_sin_1075_ood['test_losses']))],
    ["1,0.5", round(np.mean(data_con_105['test_losses'])), round(np.mean(data_sin_105['test_losses'])), round(np.mean(data_con_105_ood['test_losses'])), round(np.mean(data_sin_105_ood['test_losses']))],
    ["1,0.25", round(np.mean(data_con_1025['test_losses'])), round(np.mean(data_sin_1025['test_losses'])), round(np.mean(data_con_1025_ood['test_losses'])), round(np.mean(data_sin_1025_ood['test_losses']))],
    ["1,0", round(np.mean(data_con_10['test_losses'])), round(np.mean(data_sin_10['test_losses'])), round(np.mean(data_con_10_ood['test_losses'])), round(np.mean(data_sin_10_ood['test_losses']))],
]
print(tabulate(table))