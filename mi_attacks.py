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

def apply_mi_attack(model, loaders, idx, save_path, num_attacker_train=100, seed=0, device='cpu'):

    results = {}
    results_path = os.path.join(save_path, f'mi_results_ntrain_{num_attacker_train}_randseed_{seed}.pickle')

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
    print('--------------------------------------------')


    return results


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


def attack_wrapper(mi_loaders, idx, num_attacker_train, seed, model_params):
    # datasets for MIAs

    save_path = model_params['model_path']

    print(f'Attacking {os.path.basename(save_path)} - |A|: {num_attacker_train} - S: {seed}...')

    if not af.file_exists(os.path.join(save_path, 'clf.dat')):
        print(f'Model does not exist...')
        print('--------------------------------------------')
        return

    results_path = os.path.join(save_path, f'mi_results_ntrain_{num_attacker_train}_randseed_{seed}.pickle')

    if af.file_exists(results_path):
        apply_mi_attack(None, None, None, save_path, num_attacker_train, seed, None)
    else:  
        clf = af.load_model(os.path.join(save_path, 'clf'), device)
        apply_mi_attack(clf, mi_loaders, idx, save_path, num_attacker_train=num_attacker_train, seed=seed, device=device)


    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    af.set_random_seeds()

    device = af.get_pytorch_device()    

    # attacker's test sets contain 5000 samples
    N_ATTACKER_TEST = 5000

    ds_names = ['cifar10']
    attrs = [100]

    for rep in [None, 1, 2]:
        for ds_name in ds_names:
            models_path = f'{ds_name}_models' if rep is None else f'{ds_name}_models_rep_{rep}'
            all_models = af.collect_all_models_and_results(models_path, None, True, None)
            print(f'There are {len(all_models)} models in {models_path}.')

            # load the datasets
            datasets = af.get_ds(ds_name, device)

            for seed in af.get_random_seeds():
                for num_attacker_train in attrs:
                    # N instances from train set and N from test set
                    mi_loaders, idx = take_subset_from_datasets(datasets, seed, num_attacker_train, N_ATTACKER_TEST, device=device)
                    for model_params in all_models:
                        attack_wrapper(mi_loaders, idx, num_attacker_train, seed, model_params)