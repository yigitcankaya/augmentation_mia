import numpy as np
import os

import models as m
import aux_funcs as af

import pickle

import warnings
warnings.filterwarnings("ignore")


def create_loader_from_idx(dataset, idx, shuffle):
    data =  dataset.data[idx].cpu().detach().numpy()
    labels = dataset.labels[idx].cpu().detach().numpy()
    ds = af.ManualData(data, labels, device)
    loader = af.get_loader(ds, shuffle=shuffle, device=dataset.device)

    return loader, ds

def get_loaders(datasets, seed):
    np.random.seed(seed)
    num_train = len(datasets[0].data)
    first_half = np.random.choice(num_train, size=num_train//2, replace=False)
    second_half = np.asarray([idx for idx in np.arange(num_train) if idx not in first_half])


    first_half_loader, first_half_ds = create_loader_from_idx(datasets[0], first_half, True)
    second_half_loader, second_half_ds = create_loader_from_idx(datasets[0], second_half, True)
    test_loader = af.get_loader(datasets[1], shuffle=False, batch_size=500, device=datasets[1].device)

    return (first_half_loader, second_half_loader, test_loader), (first_half_ds, second_half_ds)


def get_loaders_from_correctly_classified(model, datasets, seed, device):
    
    first_correct_idx = m.get_correctly_classified_preds(model, af.get_loader(datasets[0], shuffle=False, batch_size=500, device=datasets[0].device), device)
    second_correct_idx = m.get_correctly_classified_preds(model, af.get_loader(datasets[1], shuffle=False, batch_size=500, device=datasets[1].device), device)

    num_samples = min(len(first_correct_idx), len(second_correct_idx))

    np.random.seed(seed)
    first_idx = np.random.choice(len(first_correct_idx), size=num_samples, replace=False)
    second_idx = np.random.choice(len(second_correct_idx), size=num_samples, replace=False)

    first_loader, _ = create_loader_from_idx(datasets[0], first_idx, True)
    second_loader, _ = create_loader_from_idx(datasets[1], second_idx, True)

    return first_loader, second_loader
    

def train_teacher(loaders, ds_name, seed, label_smoothing_alpha):
    first_half_loader, _, test_loader = loaders 

    models_path = f'smoothing_models_{ds_name}'
    af.create_path(models_path)
    
    is_dp = False

    # teacher params
    t_cap, t_train_length = 1, 'regular'

    save_func = None

    teacher_path = f'smooth_distillation_teacher_alpha_{label_smoothing_alpha}_randseed_{seed}'

    # train the teacher model with smoothing
    if af.file_exists(os.path.join(models_path, teacher_path, 'clf.dat')):
        print(f'{teacher_path} exists')
        teacher = af.load_model(os.path.join(models_path, teacher_path, 'clf'), device)
    else:
        teacher, epochs, milestones = af.get_ds_and_clf(ds_name, is_dp, t_cap, t_train_length, False, device)
        optim = af.get_std_optimizer(teacher, milestones=milestones, optim_type='adam')
        af.create_path(os.path.join(models_path, teacher_path))
        print(f'Training {teacher_path}...')
        m.train_clf(teacher, (first_half_loader, test_loader), optim, epochs, save_func, training_type='smooth', training_params=label_smoothing_alpha, device=device)
        # save the trained model
        af.save_model(os.path.join(models_path, teacher_path, 'clf'), teacher)

    return teacher
    

def train_students(teacher, loaders, ds_name, seed, label_smoothing_alpha, distillation_T):
    first_half_loader, second_half_loader, test_loader = loaders 

    models_path = f'smoothing_models_{ds_name}'
    af.create_path(models_path)
    is_dp = False

    # student params -- smaller model trained for a shorter amount of time
    s_cap, s_train_length = 0.5, 'xshort'

    save_func = None


    # train a student on the first half with distillation
    first_student_path = f'smooth_distillation_first_student_alpha_{label_smoothing_alpha}_randseed_{seed}_temp_{distillation_T}'
    if af.file_exists(os.path.join(models_path, first_student_path, 'clf.dat')):
        print(f'{first_student_path} exists')
        first_student = af.load_model(os.path.join(models_path, first_student_path, 'clf'), device)

    else:
        af.create_path(os.path.join(models_path, first_student_path))
        print(f'Training {first_student_path}...')
        first_student, epochs, milestones = af.get_ds_and_clf(ds_name, is_dp, s_cap, s_train_length, False, device)
        optim = af.get_std_optimizer(first_student, milestones=milestones, optim_type='adam')
        m.train_clf(first_student, (first_half_loader, test_loader), optim, epochs, save_func, training_type='distillation', training_params=(teacher, distillation_T), device=device)
        af.save_model(os.path.join(models_path, first_student_path, 'clf'), first_student)

    # train a student on the second half with distillation
    second_student_path = f'smooth_distillation_second_student_alpha_{label_smoothing_alpha}_randseed_{seed}_temp_{distillation_T}'
    if af.file_exists(os.path.join(models_path, second_student_path, 'clf.dat')):
        print(f'{second_student_path} exists')
        second_student = af.load_model(os.path.join(models_path, second_student_path, 'clf'), device)
    else:
        af.create_path(os.path.join(models_path, second_student_path))
        print(f'Training {second_student_path}...')
        second_student, epochs, milestones = af.get_ds_and_clf(ds_name, is_dp, s_cap, s_train_length, False, device)
        optim = af.get_std_optimizer(second_student, milestones=milestones, optim_type='adam')
        m.train_clf(second_student, (second_half_loader, test_loader), optim, epochs, save_func, training_type='distillation', training_params=(teacher, distillation_T), device=device)
        af.save_model(os.path.join(models_path, second_student_path, 'clf'), second_student)
    

    return first_student, second_student


def test_wrapper(clf, loader, device):
    top1, top5 = m.test_clf(clf, loader, device)
    print(f'Top1 Test accuracy: {top1:.2f}% - Top5 Test Acc: {top5:.2f}%')
    return top1


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    af.set_random_seeds()

    device = af.get_pytorch_device()
    ds_name = 'cifar100'


    results_path = f'{ds_name}_smooth_vs_distillation.pickle'

    if af.file_exists(results_path):
        with open(results_path, 'rb') as handle:
            results = pickle.load(handle)

    else:
        datasets = af.get_ds(ds_name, device)

        seeds = [0,1,2,3,4] #af.get_random_seeds()
        alphas = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        Ts = [1,5]

        teacher_accs = [[] for _ in alphas]
        first_student_w_dist_accs = [[[] for _ in Ts] for _ in alphas]
        second_student_w_dist_accs = [[[] for _ in Ts] for _ in alphas]

        for ii, seed in enumerate(seeds):
            loaders, subset_datasets = get_loaders(datasets, seed)

            for jj, alpha in enumerate(alphas):
                teacher = train_teacher(loaders, ds_name, seed, alpha)
                teacher_accs[jj].append(test_wrapper(teacher, loaders[-1], device))

                student_loaders = (*get_loaders_from_correctly_classified(teacher, subset_datasets, seed, device), loaders[-1])

                for kk, T in enumerate(Ts):
                    first_student, second_student = train_students(teacher, student_loaders, ds_name, seed, alpha, T)
                    first_student_w_dist_accs[jj][kk].append(test_wrapper(first_student, loaders[-1], device))
                    second_student_w_dist_accs[jj][kk].append(test_wrapper(second_student, loaders[-1], device))

                    del first_student, second_student

                    
        results = {}

        results['alphas'] = alphas
        results['seeds'] = seeds
        results['Ts'] = Ts
        results['teacher_accs'] = teacher_accs
        results['first_student_w_dist_accs'] = first_student_w_dist_accs
        results['second_student_w_dist_accs'] = second_student_w_dist_accs

        with open(results_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            