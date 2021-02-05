import numpy as np
import os

import models as m
import aux_funcs as af

from collect_results import collect_rad

import warnings
warnings.filterwarnings("ignore")


def train_wrapper(ds_name, models_path, label_aug_type, label_aug_param, data_aug_type, data_aug_param, dp_params):
    is_dp = dp_params is not None

    cap_ext = '' if cap == 1  else '_cap_{}'.format(cap)
    len_ext = '' if train_length == 'regular' else '_length_{}'.format(train_length)


    path = f'{ds_name}_laug_{label_aug_type}_{label_aug_param}_daug_{data_aug_type}_{data_aug_param}_dp_nc_{dp_params[0]}_nm_{dp_params[1]}{cap_ext}{len_ext}' \
        if is_dp else \
             f'{ds_name}_laug_{label_aug_type}_{label_aug_param}_daug_{data_aug_type}_{data_aug_param}_dp_nc_0_nm_0{cap_ext}{len_ext}'

    save_path = os.path.join(models_path, path)
    af.create_path(save_path)

    if af.file_exists(os.path.join(save_path, 'clf.dat')):
        # print(f'{path} exists')
        return

    print(f'Training: {models_path}/{path}...')
    clf, datasets, epochs, milestones = af.get_ds_and_clf(ds_name, is_dp, cap, train_length, True, device)
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
        loader_batch_size = 128

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
        train_func = lambda: m.train_clf(clf, loaders, optim, epochs, save_func, training_type='std', training_params=training_params, device=device)

    elif label_aug_type == 'no' and data_aug_type == 'mixup':
        second_train_loader = af.get_loader(datasets[0], shuffle=True, batch_size=loader_batch_size, device=datasets[0].device)
        train_func = lambda: m.train_clf(clf, loaders, optim, epochs, save_func, training_type='mixup', training_params=(second_train_loader, data_aug_param), device=device)

    elif label_aug_type == 'distillation':
        teacher = af.load_model(os.path.join(models_path, f'{ds_name}_laug_no_0_daug_no_0_dp_nc_0_nm_0', 'clf'), device)
        train_func = lambda: m.train_clf(clf, loaders, optim, epochs, save_func, training_type='distillation', training_params=(teacher, label_aug_param), device=device)
    
    elif label_aug_type == 'smooth':
        train_func = lambda: m.train_clf(clf, loaders, optim, epochs, save_func, training_type='smooth', training_params=label_aug_param, device=device)
    
    elif label_aug_type == 'disturblabel':
        train_func = lambda: m.train_clf(clf, loaders, optim, epochs, save_func, training_type='disturblabel', training_params=label_aug_param, device=device)
    
    train_func()

    # after training return the model to its original state, removes hooks etc.
    if is_dp: 
        priv_engine.detach()

    # save the trained model
    af.save_model(os.path.join(save_path, 'clf'), clf)

    del clf


def combination_models_wrapper(ds_name):
    wrapper = train_wrapper

    models_path = f'{ds_name}_models'

    results = collect_rad(print_results=True, aug_types=['crop'], ds_name=ds_name, rad_settings=[0.1, 0.25])
    alphas =  [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for alpha in alphas:
        for data_aug_param in [results['crop'][d]['param'] for d in results['crop']]:
            wrapper(ds_name=ds_name, models_path=models_path, label_aug_type='smooth', label_aug_param=alpha, data_aug_type='crop', data_aug_param=data_aug_param, dp_params=None)


def all_models_wrapper(ds_name):    
    wrapper = train_wrapper

    models_path = f'{ds_name}_models' if rep is None else f'{ds_name}_models_rep_{rep}'

    wrapper(ds_name=ds_name, models_path=models_path, label_aug_type='no', label_aug_param=0, data_aug_type='no', data_aug_param=0, dp_params=None)
    
    if not train_augmented:
        return

    label_aug_type = 'distillation' #SL   
    label_aug_params = [1, 2, 3, 5, 10, 15, 25, 50, 75, 100, 125, 150, 250, 500, 650, 700, 750, 850, 900, 1000]
    for label_aug_param in label_aug_params:
        wrapper(ds_name=ds_name, models_path=models_path, label_aug_type=label_aug_type, label_aug_param=label_aug_param, data_aug_type='no', data_aug_param=0, dp_params=None)

    label_aug_type = 'smooth' #LS
    label_aug_params = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.9915, 0.9925, 0.993, 0.9935, 0.994, 0.995]
    for label_aug_param in label_aug_params:
        wrapper(ds_name=ds_name, models_path=models_path, label_aug_type=label_aug_type, label_aug_param=label_aug_param, data_aug_type='no', data_aug_param=0, dp_params=None)

    label_aug_type = 'disturblabel' #DL
    label_aug_params = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.425, 0.45, 0.5, 0.525, 0.55, 0.575, 0.6, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.95, 0.975, 0.99]
    for label_aug_param in label_aug_params:
        wrapper(ds_name=ds_name, models_path=models_path, label_aug_type=label_aug_type, label_aug_param=label_aug_param, data_aug_type='no', data_aug_param=0, dp_params=None)

    data_aug_type = 'crop' #CR
    data_aug_params = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    for data_aug_param in data_aug_params:
        wrapper(ds_name=ds_name, models_path=models_path, label_aug_type='no', label_aug_param=0, data_aug_type=data_aug_type, data_aug_param=data_aug_param, dp_params=None)

    data_aug_type = 'noise' #GA
    data_aug_params = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35]
    for data_aug_param in data_aug_params:
        wrapper(ds_name=ds_name, models_path=models_path, label_aug_type='no', label_aug_param=0, data_aug_type=data_aug_type, data_aug_param=data_aug_param, dp_params=None)

    data_aug_type = 'cutout' #CO
    data_aug_params = [4, 8, 12, 16, 20, 24, 28, 32, 33, 34, 35, 36, 40, 44, 46, 48, 50, 52]
    for data_aug_param in data_aug_params:
        wrapper(ds_name=ds_name, models_path=models_path, label_aug_type='no', label_aug_param=0, data_aug_type=data_aug_type, data_aug_param=data_aug_param, dp_params=None)


    data_aug_type = 'mixup' #MU
    data_aug_params = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 'inf']
    for data_aug_param in data_aug_params:
        wrapper(ds_name=ds_name, models_path=models_path, label_aug_type='no', label_aug_param=0, data_aug_type=data_aug_type, data_aug_param=data_aug_param, dp_params=None)
    
    dp_norm_clips = 1
    dp_noises = [0.01, 0.025, 0.05, 0.075, 0.1, 0.175, 0.25, 0.375, 0.5, 0.75, 1., 1.75, 2.5, 3.75, 5., 7.5, 10.]
    batch_size = 64
    accumulation_steps = 4 
    
    for noise in dp_noises:
        dp_params = (dp_norm_clips, noise, batch_size, accumulation_steps)
        wrapper(ds_name=ds_name, models_path=models_path, label_aug_type='no', label_aug_param=0, data_aug_type='no', data_aug_param=0, dp_params=dp_params)



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    af.set_random_seeds()

    device = af.get_pytorch_device()

    ds_names = ['cifar10']

    rep = None
    train_augmented = True
    
    for rep in [None, 1, 2]:
        for ds_name in ds_names:
            train_augmented = True
            cap = 1
            train_length = 'regular'
            all_models_wrapper(ds_name)

            train_augmented = False
            train_lengths = ['xxxxxshort', 'xxxxshort', 'xxxshort', 'xxshort', 'xshort', 'short']
            for ii in train_lengths:
                cap = 1
                train_length = ii
                mode = 'train'
                all_models_wrapper(ds_name)

            train_augmented = False
            caps = [0.10, 0.25, 0.5] # capacity multiplier
            for ii in caps:
                cap = ii
                train_length = 'regular'
                all_models_wrapper(ds_name)

    combination_models_wrapper('cifar100')