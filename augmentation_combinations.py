import numpy as np
import models as m
import aux_funcs as af
import os
import pickle

from itertools import product, combinations

from mi_attacks import apply_mi_attack

import warnings
warnings.filterwarnings("ignore")

from scipy.stats import spearmanr

def get_avg_correlation(model_paths, first_model_name, second_model_name, num_attacker_train=100, seed=0):

    if first_model_name == second_model_name:
        combs = combinations(model_paths, 2)
    else:
        combs = product(model_paths, model_paths)

    correlations = []

    for path_1, path_2 in combs:
        with open( os.path.join(path_1, first_model_name, f'mi_results_ntrain_{num_attacker_train}_randseed_{seed}.pickle'), 'rb') as handle:
            first_results = pickle.load(handle)

        with open( os.path.join(path_2, second_model_name, f'mi_results_ntrain_{num_attacker_train}_randseed_{seed}.pickle'), 'rb') as handle:
            second_results = pickle.load(handle)

        correlations.append(compare_loss_rankings(first_results, second_results, False)[0])

    return np.mean(correlations)


def get_avg_correlation_sing(first_path, first_model_name, second_path, second_model_name, num_attacker_train=100, seed=0):

    with open( os.path.join(first_path, first_model_name, f'mi_results_ntrain_{num_attacker_train}_randseed_{seed}.pickle'), 'rb') as handle:
        first_results = pickle.load(handle)

    with open( os.path.join(second_path, second_model_name, f'mi_results_ntrain_{num_attacker_train}_randseed_{seed}.pickle'), 'rb') as handle:
        second_results = pickle.load(handle)

    return compare_loss_rankings(first_results, second_results, False)[0]


def compare_loss_rankings(first_model, second_model, print_details=False, tol=1e-6):        
        first_train, second_train = first_model['std_train_losses'], second_model['std_train_losses']
        coef, p = spearmanr(first_train, second_train)

        if print_details:
            print(f'Spearmans correlation coefficient: {coef:.2f}, p-value: {p:.2f}')
            print('---------------------------------------------------------------')

        return coef, p