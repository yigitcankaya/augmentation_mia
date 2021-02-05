import numpy as np
import models as m
import aux_funcs as af
import os
import time

from itertools import combinations

import warnings
warnings.filterwarnings("ignore")

def drop(orig, aug):
    return f'{((orig-aug)/orig)*100:.2f}%'



def collect_rad(training_length='regular', capacity=1, print_results=True, aug_types=None, rad_settings=[], ds_name='cifar10', num_attacker_train=100, rank_by_acc_proximity=[], models_path=None):
    aug_types = ['dp', 'distillation', 'smooth', 'disturblabel', 'crop', 'cutout', 'noise', 'mixup'] if aug_types is None else aug_types
    models_path = f'{ds_name}_models' if models_path is None else models_path

    params = {}
    params['aug_types'] = aug_types
    params['models_path'] = models_path


    seeds = af.get_random_seeds()
    all_models = [af.collect_all_models_and_results(models_path, num_attacker_train, False, seed) for seed in seeds]
    all_models = [sorted(models, key=lambda entry: entry['model_path']) for models in all_models]

    accs = []
    bests = []
    avgs = []
    for models in zip(*all_models):
        acc = np.mean([model['test_top1'] for model in models]) 
        avg = np.mean([model['avg_yeom_adv'] for model in models]) 
        best = np.mean([model['best_yeom_adv'] for model in models]) 
        if models[0]['laug'] == 'no' and models[0]['daug'] == 'no' and models[0]['dp_norm_clip'] == 0 and models[0]['capacity'] == 1 and models[0]['training_length'] == 'regular':
            baseline_acc = acc
            baseline_avg = avg
            baseline_best = best
        
        accs.append(acc)
        bests.append(best)
        avgs.append(avg)


    params['baseline_acc'] = baseline_acc
    params['baseline_avg'] = baseline_avg
    params['baseline_best'] = baseline_best

    adv_ranks = np.argsort(np.asarray([max(avg, best) for avg, best in zip(avgs, bests)]))
    acc_ranks = np.argsort(-np.asarray(accs))

    print(f'Baseline acc: {baseline_acc:.2f} - Baseline Best: {baseline_best:.2f} - Baseline Avg: {baseline_avg:.2f}')

    for aug_type in aug_types:
        params[aug_type] = {}
        params[aug_type]['max_acc'] = {}

        for acc_rank in acc_ranks:
            models = [all_models[idx][acc_rank] for idx in range(len(all_models))]

            if not (models[0]['capacity'] == capacity and models[0]['training_length'] == training_length):
                continue                    

            acc = np.mean([model['test_top1'] for model in models]) 
            avg_adv = np.mean([model['avg_yeom_adv'] for model in models]) 
            best_adv = np.mean([model['best_yeom_adv'] for model in models]) 
            epsilon = models[0]['epsilon']
            
            if models[0]['aug'] == aug_type:  
                param = models[0]['aug_param']
                params[aug_type]['max_acc']['acc'] = acc
                params[aug_type]['max_acc']['avg'] = avg_adv
                params[aug_type]['max_acc']['best'] = best_adv
                params[aug_type]['max_acc']['param'] = param
                params[aug_type]['max_acc']['path'] = models[0]['model_path']

                if print_results:
                    print(f'{ds_name} -- max acc -- Acc: {acc:.2f} -- Acc Drop: {drop(baseline_acc, acc)} - Aug {aug_type}-{param}-{epsilon:.2f} Avg: {avg_adv:.2f} -- Best: {best_adv:.2f} ')
                break
        
        for prox in rank_by_acc_proximity:
            prox_acc = baseline_acc * (1 - prox)
            prox_ranks = np.argsort(np.abs(np.asarray(accs) - prox_acc))
            params[aug_type][f'prox_{prox}'] = {}

            for prox_rank in prox_ranks:
                models = [all_models[idx][prox_rank] for idx in range(len(all_models))]

                if not (models[0]['capacity'] == capacity and models[0]['training_length'] == training_length):
                    continue                    

                acc = np.mean([model['test_top1'] for model in models]) 
                avg_adv = np.mean([model['avg_yeom_adv'] for model in models]) 
                best_adv = np.mean([model['best_yeom_adv'] for model in models]) 
                epsilon = models[0]['epsilon']
                
                if models[0]['aug'] == aug_type:  
                    param = models[0]['aug_param']
                    params[aug_type][f'prox_{prox}']['acc'] = acc
                    params[aug_type][f'prox_{prox}']['avg'] = avg_adv
                    params[aug_type][f'prox_{prox}']['best'] = best_adv
                    params[aug_type][f'prox_{prox}']['param'] = param
                    params[aug_type][f'prox_{prox}']['path'] = models[0]['model_path']

                    if print_results:
                        print(f'{ds_name} -- prox {prox} -- Acc: {acc:.2f} -- Acc Drop: {drop(baseline_acc, acc)} - Aug {aug_type}-{param}-{epsilon:.2f} Avg: {avg_adv:.2f} -- Best: {best_adv:.2f} ')
                    break

        for rad_setting in rad_settings:
            params[aug_type][f'rad_{rad_setting}'] = {}
            for adv_rank in adv_ranks:
                models = [all_models[idx][adv_rank] for idx in range(len(all_models))]

                if not (models[0]['capacity'] == capacity and models[0]['training_length'] == training_length):
                    continue                    

                acc = np.mean([model['test_top1'] for model in models]) 
                avg_adv = np.mean([model['avg_yeom_adv'] for model in models]) 
                best_adv = np.mean([model['best_yeom_adv'] for model in models]) 
                epsilon = models[0]['epsilon']
                
                if models[0]['aug'] == aug_type and acc > (1 - rad_setting)*baseline_acc: 
                    param = models[0]['aug_param']
                    params[aug_type][f'rad_{rad_setting}']['acc'] = acc
                    params[aug_type][f'rad_{rad_setting}']['avg'] = avg_adv
                    params[aug_type][f'rad_{rad_setting}']['best'] = best_adv
                    params[aug_type][f'rad_{rad_setting}']['param'] = param
                    params[aug_type][f'rad_{rad_setting}']['path'] = models[0]['model_path']

                    if print_results:
                        print(f'{ds_name} -- rad {rad_setting} -- Acc: {acc:.2f} -- Acc Drop: {drop(baseline_acc, acc)} - Aug {aug_type}-{param}-{epsilon:.2f} Avg: {avg_adv:.2f} -- Best: {best_adv:.2f} ')
                    break
    
    return  params



def collect_all(training_length='regular', capacity=1, laug_type='smooth', daug_type='crop', ds_name='cifar100', num_attacker_train=100, models_path=None):

    params = {}

    params['daug_type'] = daug_type
    params['laug_type'] = laug_type

    params['daug_params'] = []

    models_path = f'{ds_name}_models' if models_path is None else models_path

    params['models_path'] = models_path

    seeds = af.get_random_seeds()
    all_models = [af.collect_all_models_and_results(models_path, num_attacker_train, False, seed) for seed in seeds]
    all_models = [sorted(models, key=lambda entry: entry['model_path']) for models in all_models]

    for model_idx in range(len(all_models[0])):
        models = [all_models[idx][model_idx] for idx in range(len(all_models))]

        if not (models[0]['capacity'] == capacity and models[0]['training_length'] == training_length):
            continue                    

        acc = np.mean([model['test_top1'] for model in models]) 
        avg_adv = np.mean([model['avg_yeom_adv'] for model in models]) 
        best_adv = np.mean([model['best_yeom_adv'] for model in models]) 


        if models[0]['laug'] == laug_type and models[0]['daug'] == daug_type:
            laug_param, daug_param = models[0]['laug_param'], models[0]['daug_param']

            if daug_param not in params:
                params[daug_param] = {}
                params[daug_param]['acc'] = []
                params[daug_param]['best'] = []
                params[daug_param]['laug_param'] = []
                params[daug_param]['avg'] = []
                params['daug_params'].append(daug_param)
                params[daug_param]['path'] = []

            params[daug_param]['acc'].append(acc)
            params[daug_param]['avg'].append(avg_adv)
            params[daug_param]['best'].append(best_adv)
            params[daug_param]['laug_param'].append(laug_param)
            params[daug_param]['path'].append(models[0]['model_path'])

    return  params


def collect_dp(training_length='regular', capacity=1, ds_name='cifar100', num_attacker_train=100):

    params = {}

    models_path = f'{ds_name}_models'

    seeds = af.get_random_seeds()
    all_models = [af.collect_all_models_and_results(models_path, num_attacker_train, False, seed) for seed in seeds]
    all_models = [sorted(models, key=lambda entry: entry['model_path']) for models in all_models]

    for model_idx in range(len(all_models[0])):
        models = [all_models[idx][model_idx] for idx in range(len(all_models))]

        if not (models[0]['capacity'] == capacity and models[0]['training_length'] == training_length):
            continue                    

        acc = np.mean([model['test_top1'] for model in models]) 
        avg_adv = np.mean([model['avg_yeom_adv'] for model in models]) 
        best_adv = np.mean([model['best_yeom_adv'] for model in models]) 

        dp_noise = models[0]['dp_noise']

        if models[0]['laug'] == 'no' and models[0]['daug'] == 'no' and dp_noise != 0:
            if dp_noise not in params:
                params[dp_noise] = {}

            params[dp_noise]['acc'] = acc
            params[dp_noise]['best'] = avg_adv
            params[dp_noise]['avg'] = best_adv
            params[dp_noise]['path'] = models[0]['model_path']

    return  params