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

def get_pairwise_lrc(first_paths, second_paths, n_attacker_train=100, seed=0):

    if first_paths == second_paths:
        pairs = combinations(first_paths, 2)
    else:
        pairs = product(first_paths, second_paths)

    lrc_scores = []

    for fpath, spath in pairs:
        with open( os.path.join(fpath, f'mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle'), 'rb') as handle:
            first_results = pickle.load(handle)

        with open(os.path.join(spath, f'mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle'), 'rb') as handle:
            second_results = pickle.load(handle)

        lrc_scores.append(get_lrc_score(first_results, second_results, False)[0])
    
    return np.mean(lrc_scores)


def get_lrc_score(first_model, second_model, print_details=False, tol=1e-6):        
        first_train, second_train = first_model['std_train_losses'], second_model['std_train_losses']
        coef, p = spearmanr(first_train, second_train)

        if print_details:
            print(f'Spearmans correlation coefficient: {coef:.2f}, p-value: {p:.2f}')
            print('---------------------------------------------------------------')

        return coef, p