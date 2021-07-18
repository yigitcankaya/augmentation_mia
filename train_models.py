import numpy as np
import os

import models as m
import aux_funcs as af

import warnings
warnings.filterwarnings("ignore")


def train_wrapper(ds_name, models_path, label_aug_type, label_aug_param, data_aug_type, data_aug_param, dp_params):
    is_dp = dp_params is not None

    path = f'{ds_name}_laug_{label_aug_type}_{label_aug_param}_daug_{data_aug_type}_{data_aug_param}_dp_nc_{dp_params[0]}_nm_{dp_params[1]}_epochs_{num_epoch}_run_{path_suffix}' \
        if is_dp else \
             f'{ds_name}_laug_{label_aug_type}_{label_aug_param}_daug_{data_aug_type}_{data_aug_param}_dp_nc_0_nm_0_epochs_{num_epoch}_run_{path_suffix}'

    save_path = os.path.join(models_path, path)
    af.create_path(save_path)

    if af.file_exists(os.path.join(save_path, 'clf.dat')):
        print(f'{path} exists')
        return

    print(f'Training: {path}...')
    clf, datasets, milestones = af.get_ds_and_clf(ds_name, is_dp, num_epoch, True, device)
    optim = af.get_std_optimizer(clf, milestones=milestones, optim_type='adam')

    if is_dp:
        norm_clip, noise_mult, batch_size, accumulation_steps = dp_params
        training_size = datasets[0].data.shape[0]
        # monkey patches the optimizer
        priv_engine = af.modify_optim_for_dp(clf, optim[0], norm_clip, noise_mult, batch_size, accumulation_steps, training_size)
        clf.is_dp = True
        training_params = accumulation_steps
        loader_batch_size = batch_size
    else:
        clf.is_dp = False
        training_params = None
        loader_batch_size = cfg['regular_batch_size']

    save_func = None


    if data_aug_type == 'crop':
        datasets[0].add_crop(data_aug_param)

    if data_aug_type == 'noise':
        datasets[0].add_gaussian_aug(data_aug_param)

    if data_aug_type == 'cutout':
        datasets[0].add_cutout(data_aug_param)

    train_loader = af.get_loader(datasets[0], shuffle=True, batch_size=loader_batch_size, device=datasets[0].device)
    test_loader = af.get_loader(datasets[1], shuffle=False, batch_size=loader_batch_size, device=datasets[1].device)

    loaders = (train_loader, test_loader)

    if label_aug_type == 'no' and 'mixup' not in data_aug_type:
        train_func = lambda: m.train_clf(clf, loaders, optim, num_epoch, save_func, training_type='std', training_params=training_params, device=device)

    elif label_aug_type == 'no' and data_aug_type == 'mixup':
        second_train_loader = af.get_loader(datasets[0], shuffle=True, batch_size=loader_batch_size, device=datasets[0].device)
        train_func = lambda: m.train_clf(clf, loaders, optim, num_epoch, save_func, training_type='mixup', training_params=(second_train_loader, data_aug_param), device=device)

    elif label_aug_type == 'distillation':
        teacher = af.load_model(os.path.join(models_path, f'{ds_name}_laug_no_0_daug_no_0_dp_nc_0_nm_0_epochs_{regular_train_epochs}_run_{path_suffix}', 'clf'), device)
        train_func = lambda: m.train_clf(clf, loaders, optim, num_epoch, save_func, training_type='distillation', training_params=(teacher, label_aug_param), device=device)
    
    elif label_aug_type == 'smooth':
        train_func = lambda: m.train_clf(clf, loaders, optim, num_epoch, save_func, training_type='smooth', training_params=label_aug_param, device=device)
    
    elif label_aug_type == 'disturblabel':
        train_func = lambda: m.train_clf(clf, loaders, optim, num_epoch, save_func, training_type='disturblabel', training_params=label_aug_param, device=device)
    
    train_func()

    # after training return the model to its original state, removes hooks etc.
    if is_dp: 
        priv_engine.detach()

    # save the trained model
    af.save_model(os.path.join(save_path, 'clf'), clf)

    del clf


# training random augmentation models with label smoothing for section 5 of the paper
def combination_models_wrapper():
    crop_params = cfg['augmentation_params']['crop']
    alphas =  cfg['augmentation_params']['smooth']

    for alpha in alphas:
        for crop_param in crop_params:
            wrapper(label_aug_type='smooth', label_aug_param=alpha, data_aug_type='crop', data_aug_param=crop_param, dp_params=None)


def all_models_wrapper():    
    # baseline models
    wrapper(laug_type='no', laug_param=0, daug_type='no', daug_param=0, dp_params=None)
    
    if only_baseline:
        return

    training_augmentations = cfg['training_augmentations']

    if 'distillation' in training_augmentations: #SL
        params = cfg['augmentation_params']['distillation']
        for laug_param in params:
            wrapper(laug_type='distillation', laug_param=laug_param, daug_type='no', daug_param=0, dp_params=None)

    if 'smooth' in training_augmentations: #LS
        params = cfg['augmentation_params']['smooth']
        for laug_param in params:
            wrapper(laug_type='smooth', laug_param=laug_param, daug_type='no', daug_param=0, dp_params=None)

    if 'disturblabel' in training_augmentations: #DL
        params = cfg['augmentation_params']['disturblabel']
        for laug_param in params:
            wrapper(laug_type='disturblabel', laug_param=laug_param, daug_type='no', daug_param=0, dp_params=None)

    if 'crop' in training_augmentations: #RC
        params = cfg['augmentation_params']['crop']
        for daug_param in params:
            wrapper(laug_type='no', laug_param=0, daug_type='crop', daug_param=daug_param, dp_params=None)

    if 'noise' in training_augmentations: #GA
        params = cfg['augmentation_params']['noise']
        for daug_param in params:
            wrapper(laug_type='no', laug_param=0, daug_type='noise', daug_param=daug_param, dp_params=None)

    if 'cutout' in training_augmentations: #CO
        params = cfg['augmentation_params']['cutout']
        for daug_param in params:
            wrapper(laug_type='no', laug_param=0, daug_type='cutout', daug_param=daug_param, dp_params=None)

    if 'mixup' in training_augmentations: #MU
        params = cfg['augmentation_params']['mixup']
        for daug_param in params:
            wrapper(laug_type='no', laug_param=0, daug_type='mixup', daug_param=daug_param, dp_params=None)

    if cfg['train_with_dp']:
        norm_clip = cfg['dp_params']['norm_clip']
        noise_coeffs = cfg['dp_params']['noise_coeffs']
        batch_size = cfg['dp_params']['batch_size']
        accumulation_steps = cfg['dp_params']['accumulation_steps'] 
        
        for noise_coeff in noise_coeffs:
            dp_params = (norm_clip, noise_coeff, batch_size, accumulation_steps)
            wrapper(label_aug_type='no', label_aug_param=0, data_aug_type='no', data_aug_param=0, dp_params=dp_params)



if __name__ == "__main__":

    import json
    import sys

    config_path = sys.argv[1]

    with open(config_path) as f:
        cfg = json.load(f)

    af.set_random_seeds()

    device = af.get_pytorch_device()

    
    ds_names = cfg['training_datasets']
    path_suffices = cfg['path_suffices'] # for training multiple models
    regular_train_epochs = cfg['training_num_epochs']
    early_stopping_epochs = cfg['early_stopping_epochs']
    models_path = cfg['models_path']

    for path_suffix in path_suffices: # for all path suffix

        for ds_name in ds_names: # for all datasets 
            
            cur_path = af.create_path(os.path.join(models_path, ds_name)) # create a path for the trained models on this dataset

            # a wrapper around the main training function for this dataset
            wrapper = lambda laug_type, laug_param, daug_type, daug_param, dp_params: train_wrapper(ds_name, cur_path, laug_type, laug_param, daug_type, daug_param, dp_params)
            
            # train all models with specified augmentations 
            only_baseline = False
            num_epoch = regular_train_epochs
            all_models_wrapper()

            if cfg["train_early_stopping_models"]:
                for num_epoch in early_stopping_epochs: # for all training lengths
                    only_baseline = True # train non-augmented models for baselines
                    all_models_wrapper()

    if cfg['train_crop_smooth']:
        crop_smooth_dataset = 'cifar100' # in the paper, we only consider cropping+label smoothing for CIFAR-100
        for path_suffix in path_suffices: # for all path suffix
            combination_models_wrapper(crop_smooth_dataset)