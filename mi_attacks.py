import numpy as np
import pickle
import os

import models as m
import aux_funcs as af

import warnings
warnings.filterwarnings("ignore")



def split_indices(indices, first_split_size):
    first_split_indices = np.random.choice(indices, size=first_split_size, replace=False)
    second_split_indices = np.array([x for x in indices if x not in first_split_indices])
    
    return first_split_indices, second_split_indices

def apply_avg_and_best_attacks(train_losses, test_losses, idx):
    train_in_atk_train_idx, train_in_atk_test_idx, test_in_atk_train_idx, test_in_atk_test_idx = idx

    avg_loss_train = np.mean(train_losses[train_in_atk_train_idx])
    avg_train_memberships = yeom_mi_attack(train_losses[train_in_atk_test_idx], avg_loss_train)
    avg_test_memberships = yeom_mi_attack(test_losses[test_in_atk_test_idx], avg_loss_train)
    avg_yeom_mi_advantage = mi_success(avg_train_memberships, avg_test_memberships, print_details=False)

    avg_results = (avg_loss_train, avg_train_memberships, avg_test_memberships, avg_yeom_mi_advantage)

    best_threshold = yeom_w_get_best_threshold(train_losses[train_in_atk_train_idx], test_losses[test_in_atk_train_idx])
    best_train_memberships = yeom_mi_attack(train_losses[train_in_atk_test_idx], best_threshold)
    best_test_memberships = yeom_mi_attack(test_losses[test_in_atk_test_idx], best_threshold)
    best_yeom_mi_advantage = mi_success(best_train_memberships, best_test_memberships, print_details=False)

    best_results = (best_threshold, best_train_memberships, best_test_memberships, best_yeom_mi_advantage)

    return avg_results, best_results


def take_subset_from_datasets(datasets, seed, n_attacker_train, n_attacker_test, batch_size=1000, device='cpu'):

    np.random.seed(seed)
    train_indices = np.random.choice(len(datasets[0].data), size=n_attacker_train + n_attacker_test, replace=False)
    test_indices = np.random.choice(len(datasets[1].data), size=n_attacker_train + n_attacker_test, replace=False)

    train_in_atk_test_idx, train_in_atk_train_idx = split_indices(train_indices, n_attacker_test)
    test_in_atk_test_idx, test_in_atk_train_idx = split_indices(test_indices, n_attacker_test)

    train_data = datasets[0].data[np.concatenate((train_in_atk_train_idx, train_in_atk_test_idx))].cpu().detach().numpy()
    train_labels = datasets[0].labels[np.concatenate((train_in_atk_train_idx, train_in_atk_test_idx))].cpu().detach().numpy()

    test_data = datasets[1].data[np.concatenate((test_in_atk_train_idx, test_in_atk_test_idx))].cpu().detach().numpy()
    test_labels = datasets[1].labels[np.concatenate((test_in_atk_train_idx, test_in_atk_test_idx))].cpu().detach().numpy()

    train_ds = af.ManualData(train_data, train_labels)
    train_ds.train = False

    test_ds = af.ManualData(test_data, test_labels)
    test_ds.train = False

    train_loader = af.get_loader(train_ds, shuffle=False, batch_size=batch_size, device=device)
    test_loader = af.get_loader(test_ds, shuffle=False, batch_size=batch_size, device=device)

    train_in_atk_train_idx, train_in_atk_test_idx = np.arange(len(train_in_atk_train_idx)), np.arange(len(train_in_atk_train_idx), len(train_data))
    test_in_atk_train_idx, test_in_atk_test_idx = np.arange(len(test_in_atk_train_idx)), np.arange(len(test_in_atk_train_idx), len(test_data))

    idx = (train_in_atk_train_idx, train_in_atk_test_idx, test_in_atk_train_idx, test_in_atk_test_idx)
    
    return (train_loader, test_loader), idx


def apply_mi_attack(model, loaders, idx, save_path, n_attacker_train=100, seed=0, device='cpu'):

    results = {}
    results_path = os.path.join(save_path, f'mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle')

    if af.file_exists(results_path):
        with open(results_path, 'rb') as handle:
            results = pickle.load(handle)
    
    else:
        train_top1, train_top5 = m.test_clf(model, loaders[0], device)
        test_top1, test_top5 = m.test_clf(model, loaders[1], device)
        
        train_losses = m.get_clf_losses(model, loaders[0], device=device)
        test_losses = m.get_clf_losses(model, loaders[1], device=device)

        # apply vanilla yeom attacks
        avg_results, best_results = apply_avg_and_best_attacks(train_losses, test_losses, idx)
        avg_loss_train, avg_train_memberships, avg_test_memberships, avg_yeom_adv = avg_results
        best_threshold, best_train_memberships, best_test_memberships, best_yeom_adv = best_results

        results['train_top1'], results['train_top5'], results['test_top1'], results['test_top5'] = train_top1, train_top5, test_top1, test_top5
        results['avg_yeom_adv'], results['best_yeom_adv'], results['avg_threshold'], results['best_threshold'] = avg_yeom_adv, best_yeom_adv, avg_loss_train, best_threshold
        results['avg_train_memberships'], results['avg_test_memberships'] = avg_train_memberships, avg_test_memberships
        results['best_train_memberships'], results['best_test_memberships'] = best_train_memberships, best_test_memberships
        results['std_train_losses'], results['std_test_losses'] = train_losses, test_losses
        results['attack_idx'] = idx

        with open(results_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('Train Top1: {0:.3f}%, Train Top5: {1:.3f}%, Test Top1: {2:.3f}%, Test Top5: {3:.3f}%'.format(results['train_top1'], results['train_top5'], results['test_top1'], results['test_top5']))
    print('Avg Yeom MI Advantage: {0:.2f}'.format(results['avg_yeom_adv']))
    print('Best Yeom MI Advantage: {0:.2f}'.format(results['best_yeom_adv']))

    return results


# apply the augmentation aware attacks, n_repeat is for random augmentation methods
def apply_aware_attack(model, params, loaders, idx, save_path, n_attacker_train=100, n_repeat=25, seed=0, teacher=None, device='cpu'):

    results = {}
    results_path = os.path.join(save_path, f'aware_mi_results_ntrain_{n_attacker_train}_numrepeat_{n_repeat}_randseed_{seed}.pickle')

    if af.file_exists(results_path):
        with open(results_path, 'rb') as handle:
            results = pickle.load(handle)

    else:
        laug_type, laug_param = params['laug_type'], params['laug_param']
        daug_type, daug_param = params['daug_type'], params['daug_param']

        train_in_atk_train_idx, _, test_in_atk_train_idx, _ = idx

        if daug_type == 'mixup':
            mixing_data = loaders[0].dataset.data[train_in_atk_train_idx].to(device) # use attackers training data in the victim's training set for mixup
            mixing_labels = loaders[0].dataset.labels[train_in_atk_train_idx].to(device)
            aug_type, aug_param = daug_type, (daug_param, mixing_data, mixing_labels)

        elif laug_type == 'distillation':
            aug_type, aug_param = laug_type, (laug_param, teacher)

        elif laug_type != 'no':
            aug_type, aug_param = laug_type, laug_param
        
        elif daug_type != 'no':
            aug_type, aug_param = daug_type, daug_param
        

        train_losses = m.get_clf_losses_w_aug(model, loaders[0], aug_type, aug_param, num_repeat=n_repeat, device=device)
        test_losses = m.get_clf_losses_w_aug(model, loaders[1], aug_type, aug_param, num_repeat=n_repeat, device=device)

        if n_repeat == 1:
            _, aware_results = apply_avg_and_best_attacks(train_losses, test_losses, idx)
            threshold, train_memberships, test_memberships, adv = aware_results
            reduction = 'none'
            train_losses_all = train_losses
            test_losses_all = test_losses

        else:
            best_local_adv = -100

            for name, func in zip(*af.get_reduction_params()):
                cur_train_losses, cur_test_losses = func(train_losses, axis=1), func(test_losses, axis=1)

                _, aware_results = apply_avg_and_best_attacks(cur_train_losses, cur_test_losses, idx)
                cur_threshold, cur_train_memberships, cur_test_memberships, cur_adv = aware_results
                adv_local = mi_success(yeom_mi_attack(cur_train_losses[train_in_atk_train_idx], cur_threshold), yeom_mi_attack(cur_test_losses[test_in_atk_train_idx], cur_threshold), False)
                if best_local_adv < adv_local:
                    best_local_adv = adv_local
                    adv = cur_adv
                    train_memberships = cur_train_memberships
                    test_memberships = cur_test_memberships
                    train_losses_all = cur_train_losses
                    test_losses_all = cur_test_losses
                    threshold = cur_threshold
                    reduction = name


        results['threshold'], results['adv'] = threshold, adv
        results['train_memberships'], results['test_memberships'] = train_memberships, test_memberships
        results['train_losses'], results['test_losses'] = train_losses_all, test_losses_all
        results['num_repeat'] = n_repeat
        results['reduction'] = reduction

        with open(results_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Aware MI Advantage: {0:.2f} - Reduction: {1}'.format(results['adv'], results['reduction']))




def mi_success(train_memberships, test_memberships, print_details=True):
    tp = np.sum(train_memberships)
    fp = np.sum(test_memberships)
    fn = len(train_memberships) - tp
    tn = len(test_memberships) - fp

    # yeom's membership inference advantage
    acc = 100*(tp + tn) / (tp + fp + tn + fn)
    advantage = 2*(acc - 50)

    if print_details:
        precision = 100*(tp/(tp+fp)) if (tp+fp) > 0 else 0
        recall = 100*(tp/(tp+fn)) if (tp+fn) > 0 else 0

        print('Adversary Advantage: {0:.3f}%, Accuracy: {1:.3f}%, Precision : {2:.3f}%, Recall: {3:.3f}%'.format(advantage,  acc, precision, recall))
        print('In training: {}/{}, In testing: {}/{}'.format(tp, len(train_memberships), tn, len(test_memberships)))

    return advantage

# YEOM et all's membership inference attack using pred loss
def yeom_mi_attack(losses, avg_loss):
    memberships = (losses < avg_loss).astype(int)
    return memberships


def yeom_w_get_best_threshold(train_losses, test_losses):    
    advantages = []

    mean_loss = np.mean(train_losses)
    std_dev = np.std(train_losses)

    coeffs = np.linspace(-5,5,num=1001, endpoint=True)

    for coeff in coeffs:
        cur_threshold = mean_loss + std_dev*coeff
        cur_yeom_mi_advantage = mi_success(yeom_mi_attack(train_losses, cur_threshold),  yeom_mi_attack(test_losses, cur_threshold), print_details=False)
        advantages.append(cur_yeom_mi_advantage)

    best_threshold = mean_loss + std_dev*coeffs[np.argmax(advantages)]

    return best_threshold


def attack_wrapper(mi_loaders, idx, n_attacker_train, seed, params):
    # datasets for MIAs

    save_dir = params['dir']

    print(f'Attacking {os.path.basename(save_dir)} - |A|: {n_attacker_train} - S: {seed}...')


    results_path = os.path.join(save_dir, f'mi_results_ntrain_{n_attacker_train}_randseed_{seed}.pickle')

    if af.file_exists(results_path):
        apply_mi_attack(None, None, None, save_dir, n_attacker_train, seed, None)
    else:  
        clf = af.load_model(params['model_path'], device)
        apply_mi_attack(clf, mi_loaders, idx, save_dir, n_attacker_train=n_attacker_train, seed=seed, device=device)


    if params['laug_type'] != 'no' or params['daug_type'] != 'no':
        num_repeat = 1 if params['laug_type'] in ['distillation', 'smooth'] else n_repeat
        results_path = os.path.join(save_dir, f'aware_mi_results_ntrain_{n_attacker_train}_numrepeat_{n_repeat}_randseed_{seed}.pickle')

        if af.file_exists(results_path):
            apply_aware_attack(None, None, None, None, save_dir, n_attacker_train, n_repeat, seed, None, None)
        else:  
            if params['laug_type'] == 'distillation':
                teacher_dir = os.path.dirname(save_dir)
                path_suffix = params['path_suffix']
                teacher_path = f'{params["dset_name"]}_laug_no_0_daug_no_0_dp_nc_0_nm_0_epochs_{regular_train_epochs}_run_{path_suffix}'
                teacher = af.load_model(os.path.join(teacher_dir, teacher_path, 'clf'), device)
            else:
                teacher = None

            clf = af.load_model(os.path.join(save_dir, 'clf'), device)
            apply_aware_attack(clf, params, mi_loaders, idx, save_dir, n_attacker_train=n_attacker_train, n_repeat=num_repeat, seed=seed, teacher=teacher, device=device)

    print('--------------------------------------------')


    
if __name__ == "__main__":

    af.set_random_seeds()

    import json
    import sys

    config_path = sys.argv[1]

    with open(config_path) as f:
        cfg = json.load(f)


    n_attacker_train = cfg['attack']['n_attacker_train']
    n_attacker_test = cfg['attack']['n_attacker_test']
    seeds = cfg['attack']['random_seeds']
    n_repeat = cfg['attack']['n_aware_repeat']
    ds_names = cfg['training_datasets']
    models_path = cfg['models_path']

    device = af.get_pytorch_device()    

    regular_train_epochs = cfg['training_num_epochs']


    for ds_name in ds_names:
        path = os.path.join(models_path, ds_name)
        all_model_params = af.collect_all_models(path)

        print(f'There are {len(all_model_params)} models in {path}.')

        # load the datasets
        datasets = af.get_ds(ds_name, device)

        for seed in af.get_random_seeds():
            # N instances from train set and N from test set
            mi_loaders, idx = take_subset_from_datasets(datasets, seed, n_attacker_train, n_attacker_test, device=device)
            for params in all_model_params:
                attack_wrapper(mi_loaders, idx, n_attacker_train, seed, params)