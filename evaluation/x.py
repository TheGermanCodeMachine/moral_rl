import pickle as pkl
import os
import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import torch
import numpy as np


def combine_data():
    with open('datasets\\1000_ablations\weights\\set1\\100\\results_sidebysideLM\80\data_split.pkl', 'rb') as f:
        train_set1, train_labels1, test_set1, test_labels1 = pkl.load(f)
    with open('datasets\\1000_ablations\weights\set2\\100\\results_sidebysideLM\80\data_split.pkl', 'rb') as f: 
        train_set2, train_labels2, test_set2, test_labels2 = pkl.load(f)
    with open('datasets\\1000_ablations\weights\set3\\100\\results_sidebysideLM\80\data_split.pkl', 'rb') as f:
        train_set3, train_labels3, test_set3, test_labels3 = pkl.load(f)
    with open('datasets\\1000_ablations\weights\set4\\100\\results_sidebysideLM\80\data_split.pkl', 'rb') as f:
        train_set4, train_labels4, test_set4, test_labels4 = pkl.load(f)
    with open('datasets\\1000_ablations\weights\\set5\\100\\results_sidebysideLM\80\data_split.pkl', 'rb') as f:
        train_set5, train_labels5, test_set5, test_labels5 = pkl.load(f)
    # with open('datasets\\1000_ablations\only_one\only_sparsity\\100\\results_sidebysideLM\80\data_split.pkl', 'rb') as f:
        # train_set6, train_labels6, test_set6, test_labels6 = pkl.load(f)

    # append training data together
    test_set = torch.cat((test_set1, test_set2, test_set3, test_set4, test_set5), dim=0)
    test_labels = torch.cat((test_labels1, test_labels2, test_labels3, test_labels4, test_labels5), dim=0)
    with open('datasets\\1000_ablations\weights\combined_test_set.pkl', 'wb') as f:
        pkl.dump((test_set, test_labels), f)

combine_data()