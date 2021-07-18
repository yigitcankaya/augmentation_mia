import numpy as np
import models as m
import aux_funcs as af
import os
import time
import pickle

from itertools import combinations

import warnings
warnings.filterwarnings("ignore")


def get_models_dirs(models_path, ds_name, laug_type, laug_param, daug_type, daug_param, num_epochs, dp_params=None):
    all_model_params = af.collect_all_models(os.path.join(models_path, ds_name))
    
    dirs = []

    for params in all_model_params:

        if dp_params is not None and \
            params['laug_type'] == 'no' and params['daug_type'] == 'no' and \
            params['dp'] and params['dp_norm_clip'] == dp_params[0] and params['dp_noise'] == dp_params[1]:

            dir = params['dir']
            dirs.append(dir)

        elif dp_params is None and \
            params['laug_type'] == laug_type and params['laug_param'] == laug_param and \
            params['daug_type'] == daug_type and params['daug_param'] == daug_param and \
            not params['dp'] and params['num_epochs'] == num_epochs:

            dir = params['dir']
            dirs.append(dir)

    return dirs

def get_non_aug_stats(models_path, ds_name, n_attacker_train, num_epochs):
    all_model_params = af.collect_all_models(os.path.join(models_path, ds_name))
    result_files = []

    for params in all_model_params:
        if params['laug_type'] == 'no' and params['daug_type'] == 'no' and params['dp_norm_clip'] == 0 and params['num_epochs'] == num_epochs:
            dir = params['dir']

            for fn in os.listdir(dir):
                if  f'mi_results_ntrain_{n_attacker_train}' in fn:
                    result_files.append(os.path.join(dir, fn))

    
    accs = []
    avg_mias = []
    pow_mias = []

    for fn in result_files:
        with open(fn, 'rb') as handle:
            results = pickle.load(handle)

        accs.append(results['test_top1'])
        avg_mias.append(results['avg_yeom_adv'])
        pow_mias.append(results['best_yeom_adv'])

    return {'acc':np.mean(accs), 'avg_mia':np.mean(avg_mias), 'pow_mia':np.mean(pow_mias)}


def get_mia_stats(models_path, ds_name, laug_type, daug_type, n_attacker_train, n_repeat, num_epochs, collect_dp=False, sort_order='mia'):
    all_model_params = af.collect_all_models(os.path.join(models_path, ds_name))
    
    dirs = []

    for params in all_model_params:

        if collect_dp and params['dp'] and params['laug_type'] == laug_type and params['daug_type'] == daug_type and params['num_epochs'] == num_epochs:
            dir = params['dir']
            dirs.append(dir)

        elif not collect_dp and params['laug_type'] == laug_type and params['daug_type'] == daug_type and not params['dp'] and params['num_epochs'] == num_epochs:
            dir = params['dir']
            dirs.append(dir)

    dir_prefixes = set([''.join(d.split('_')[:-1]) for d in dirs])
    
    # each dir group contains the models trained with same parameters (different runs)
    dir_groups = [[d for d in dirs if ''.join(d.split('_')[:-1])==pf] for pf in dir_prefixes]
    all_results = []

    for dir_group in dir_groups:
        
        cur_results = {}

        params = af.parse_model_path(os.path.basename(dir_group[0]))

        result_files = []
        aware_result_files = []

        for dir in dir_group:
            for fn in os.listdir(dir):
                if  f'mi_results_ntrain_{n_attacker_train}' in fn and 'aware' not in fn:
                    result_files.append(os.path.join(dir, fn))
                
                elif f'aware_mi_results_ntrain_{n_attacker_train}_numrepeat_{n_repeat}' in fn:
                    aware_result_files.append(os.path.join(dir, fn))

                elif f'aware_mi_results_ntrain_{n_attacker_train}_numrepeat_1' in fn:
                    aware_result_files.append(os.path.join(dir, fn))

        accs = []
        avg_mias = []
        pow_mias = []
        for fn in result_files:
            with open(fn, 'rb') as handle:
                results = pickle.load(handle)
            
            accs.append(results['test_top1'])
            avg_mias.append(results['avg_yeom_adv'])
            pow_mias.append(results['best_yeom_adv'])

        aware_mias = []

        for fn in aware_result_files:
            with open(fn, 'rb') as handle:
                results = pickle.load(handle)

            aware_mias.append(results['adv'])

        cur_results['laug_type'], cur_results['daug_type'] = laug_type, daug_type
        cur_results['laug_param'], cur_results['daug_param'] = params['laug_param'], params['daug_param']
        cur_results['acc'], cur_results['avg_mia'], cur_results['pow_mia'] = np.mean(accs), np.mean(avg_mias), np.mean(pow_mias)

        if len(aware_mias) > 0:
            cur_results['awa_mia'] = np.mean(aware_mias),  
        else:
            cur_results['awa_mia'] = cur_results['pow_mia']  

        if collect_dp:
            print(dir_group[0])
            epsilon = af.load_model(os.path.join(dir_group[0], 'clf')).dp_epsilons[-1]
            cur_results['epsilon'] = epsilon
            cur_results['dp_noise'] = params['dp_noise']
            cur_results['dp_norm_clip'] = params['dp_norm_clip']


        all_results.append(cur_results)
    
    if sort_order == 'mia':
        # get the maximum successful attack accuracy and sort it based on that
        all_results = sorted(all_results, key=lambda x: max(x['avg_mia'], x['pow_mia'], x.get('awa_mia', 'pow_mia')))
    else:
        # sort the results based on the model accuracy
        all_results = sorted(all_results, key=lambda x: -x['acc'])

    return all_results