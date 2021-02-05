import numpy as np

import os 

import torch
import torch.nn as nn
import torch.nn.functional as F

import aux_funcs as af

from tqdm import tqdm

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)



class FMNISTClassifier(nn.Module):
    def __init__(self, num_classes=10, dp=False, cap=1, device='cpu'):
        super(FMNISTClassifier, self).__init__()
        self.num_classes = num_classes
        self.image_size = 28

        if dp:
            # these record the alphas and epsilons as the network is trained
            self.dp_best_alphas = []
            self.dp_epsilons = [] 

            BN = lambda num_features : nn.GroupNorm(min(32, num_features), num_features, affine=True)
        else:
            BN = lambda num_features : nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        
        self.model = nn.Sequential(
            nn.Conv2d(1, int(128*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(128*cap)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(int(128*cap), int(256*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(256*cap)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            Flatten(),

            nn.Linear(in_features=int(256*cap)*49, out_features=1024, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=self.num_classes, bias=True),
        ).to(device)

    def forward(self, x):
        return nn.functional.log_softmax(self.model(x), dim=1)

    def forward_w_temperature(self, x, T=1):
        logits = self.model(x)
        scaled_logits = logits/T
        return nn.functional.softmax(scaled_logits, dim=1)


class CIFARClassifier(nn.Module):
    def __init__(self, num_classes=10, dp=False, cap=1, device='cpu'):
        super(CIFARClassifier, self).__init__()
        self.num_classes = num_classes
        self.image_size = 32
        
        if dp:
            # these record the alphas and epsilons as the network is trained
            self.dp_best_alphas = []
            self.dp_epsilons = [] 

            BN = lambda num_features : nn.GroupNorm(min(32, num_features), num_features, affine=True)
        else:
            BN = lambda num_features : nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        

        self.model = nn.Sequential(
            nn.Conv2d(3, int(64*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(64*cap)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(int(64*cap), int(128*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(128*cap)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(int(128*cap), int(256*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(256*cap)),
            nn.ReLU(True),

            nn.Conv2d(int(256*cap), int(256*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(256*cap)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(int(256*cap), int(512*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(512*cap)),
            nn.ReLU(True),
            
            nn.Conv2d(int(512*cap), int(512*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(512*cap)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(int(512*cap), int(512*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(512*cap)),
            nn.ReLU(True),

            nn.Conv2d(int(512*cap), int(512*cap), kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            BN(int(512*cap)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            Flatten(),            

            nn.Linear(in_features=int(512*cap), out_features=1024, bias=True),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=self.num_classes)

        ).to(device)

        
    def forward(self, x):
        return nn.functional.log_softmax(self.model(x), dim=1)
        
    def forward_w_temperature(self, x, T=1):
        logits = self.model(x)
        scaled_logits = logits/T
        return nn.functional.softmax(scaled_logits, dim=1)



def SoftLabelNLL(predicted, target, reduce=False):
    if reduce:
        return -(target * predicted).sum(dim=1).mean()
    else:
        return -(target * predicted).sum(dim=1)

def clf_std_training_step(clf, optimizer, data, labels, device='cpu'):
    clf.train()

    clf_loss_func = nn.NLLLoss()
    b_x = data.to(device, dtype=torch.float)   # batch x
    b_y = labels.to(device, dtype=torch.long)   # batch y

    clf_output = clf(b_x)
    loss = clf_loss_func(clf_output, b_y)
    
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients

    del clf_output, b_x, b_y, loss


def clf_dp_training_step(clf, optimizer, data, labels, batch_idx, accumulation_steps, tot_batches, device='cpu'):
    clf.train()

    clf_loss_func = nn.NLLLoss()
    b_x = data.to(device, dtype=torch.float)   # batch x
    b_y = labels.to(device, dtype=torch.long)   # batch y

    clf_output = clf(b_x)
    loss = clf_loss_func(clf_output, b_y)
    
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients


    # make sure we take a step after processing the last mini-batch in the
    # epoch to ensure we start the next epoch with a clean state
    if (batch_idx % accumulation_steps == 0) or (batch_idx == tot_batches):
        optimizer.step()
    else:
        optimizer.virtual_step() # to be able to use large batch sizes without needing large memory

    del clf_output, b_x, b_y, loss


def clf_mixup_training_step(clf, optimizer, data_1, labels_1, data_2, labels_2, alpha, device='cpu'):
    clf.train()

    if alpha == 'inf':
        lam = 0.5 # simple averaging
    elif alpha == 0:
        lam = 1 # no interpolation
    else:
        lam = np.random.beta(alpha, alpha)

    b_x = (lam * data_1.to(device, dtype=torch.float)) + ((1 - lam) * data_2.to(device, dtype=torch.float))   # batch x

    clf_loss_func = lambda pred, target: SoftLabelNLL(pred, target, reduce=True)
    labels_1_one_hot = ((torch.zeros(data_1.shape[0], clf.num_classes, dtype=torch.float)).to(device)).scatter_(1, labels_1.view(-1, 1), 1)
    labels_2_one_hot = ((torch.zeros(data_2.shape[0], clf.num_classes, dtype=torch.float)).to(device)).scatter_(1, labels_2.view(-1, 1), 1)
    b_y = (lam * labels_1_one_hot) + ( (1 - lam) * labels_2_one_hot)   # batch y

    clf_output = clf(b_x)
    loss = clf_loss_func(clf_output, b_y)
    
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients

    del clf_output, b_x, b_y, labels_1_one_hot, labels_2_one_hot, loss


def clf_disturblabel_training_step(clf, optimizer, data, labels, alpha, device='cpu'):
    clf.train()

    C = clf.num_classes
    p_c = (1 - ((C - 1)/C) * alpha)
    p_i = (1 / C) * alpha

    clf_loss_func = nn.NLLLoss()

    b_x = data.to(device, dtype=torch.float)   # batch x
    b_y = labels.to(device, dtype=torch.long).view(-1, 1)   # batch y

    b_y_one_hot = (torch.ones(b_y.shape[0], C) * p_i).to(device)
    b_y_one_hot.scatter_(1, b_y, p_c)
    b_y_one_hot = b_y_one_hot.view( *(tuple(labels.shape) + (-1,) ) )

    # sample from Multinoulli distribution
    distribution = torch.distributions.OneHotCategorical(b_y_one_hot)
    b_y_disturbed = distribution.sample()
    b_y_disturbed = b_y_disturbed.max(dim=1)[1]  # back to categorical

    clf_output = clf(b_x)
    loss = clf_loss_func(clf_output, b_y_disturbed)
    
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients

    del clf_output, b_x, b_y, b_y_one_hot, loss


def clf_distillation_training_step(clf, optimizer, data, labels, teacher, T, device='cpu'):
    teacher.eval()
    clf.train()

    clf_loss_func = lambda pred, target: SoftLabelNLL(pred, target, reduce=True)
    b_x = data.to(device, dtype=torch.float)   # batch x
    #b_y = labels.to(device, dtype=torch.long)   # batch y

    with torch.no_grad():
        b_y = teacher.forward_w_temperature(b_x, T)

    clf_output = clf(b_x)

    loss = clf_loss_func(clf_output, b_y)
    
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients

    del clf_output, b_x, b_y, loss


def clf_smoothlabel_training_step(clf, optimizer, data, labels, smoothing_coef, device='cpu'):
    clf.train()

    clf_loss_func = lambda pred, target: SoftLabelNLL(pred, target, reduce=True)
    b_x = data.to(device, dtype=torch.float)   # batch x
    b_y = labels.to(device, dtype=torch.long)   # batch y
    b_y_one_hot = (torch.zeros(data.shape[0], clf.num_classes, dtype=torch.float).to(device)).scatter_(1, b_y.view(-1, 1), 1)

    b_y_one_hot = (1-smoothing_coef)*b_y_one_hot + (smoothing_coef/clf.num_classes)
    
    clf_output = clf(b_x)
    loss = clf_loss_func(clf_output, b_y_one_hot)
    
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients
    
    del clf_output, b_x, b_y, b_y_one_hot, loss


def test_clf(clf, loader, device='cpu'):
    clf.eval()
   
    top1 = af.AverageMeter()
    top5 = af.AverageMeter()

    with torch.no_grad():
        for x, y in loader:
            b_x = x.to(device, dtype=torch.float)
            b_y = y.to(device, dtype=torch.long)

            clf_output = clf(b_x)

            if clf.num_classes < 5:
                accs = af.accuracy(clf_output, b_y, topk=(1, ))
            else:
                accs = af.accuracy(clf_output, b_y, topk=(1, 5))
                top5.update(accs[1], b_x.size(0))

            top1.update(accs[0], b_x.size(0))


    top1_acc = top1.avg
    if clf.num_classes < 5:
        top5_acc = 100.
    else:
        top5_acc = top5.avg

    return top1_acc, top5_acc

def get_clf_losses(clf, loader, device='cpu'):

    clf_loss_func = nn.NLLLoss(reduction='none')

    losses = np.zeros(af.loader_inst_counter(loader))
    cur_idx = 0

    clf.eval()
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            b_y = batch[1].to(device, dtype=torch.long)

            output = clf(b_x)
            losses[cur_idx:cur_idx+len(b_x)] = clf_loss_func(output, b_y).flatten().cpu().detach().numpy()
            cur_idx += len(b_x)

    
    return losses

def get_clf_preds(clf, loader, logits=True, temperature=1, device='cpu'):

    preds = np.zeros((af.loader_inst_counter(loader), clf.num_classes))
    cur_idx = 0

    clf.eval()
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            output = clf.model(b_x) if logits else clf.forward_w_temperature(b_x, T=temperature).cpu().detach().numpy()
            preds[cur_idx:cur_idx+len(b_x)] = output
            cur_idx += len(b_x)
    
    return preds


def get_correctly_classified_preds(clf, loader, device='cpu'):

    idx = []

    cur_idx = 0

    clf.eval()
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device, dtype=torch.float)
            b_y = batch[1].to(device, dtype=torch.long)
            preds = clf(b_x).max(dim=1)[1]
            correct_idx = torch.where(b_y == preds)[0].detach().cpu().numpy()
            idx.append(correct_idx + cur_idx)
            cur_idx += len(b_x)
    
    idx = np.concatenate(idx)

    return idx

def train_clf(clf, loaders, optimizer, epochs, save_func=None, training_type='std', training_params=None, device='cpu'):
    print('Clf Train')
    train_loader, test_loader = loaders
    opt, sch = optimizer

    if not hasattr(clf, 'is_dp') or (not clf.is_dp):
        clf.is_dp = False
        if training_type == 'std':
            step_func = lambda data, labels, batch_idx: clf_std_training_step(clf, opt, data, labels, device)
        
        elif training_type == 'distillation':
            teacher, T = training_params
            step_func = lambda data, labels, batch_idx: clf_distillation_training_step(clf, opt, data, labels, teacher, T, device)
        
        elif training_type == 'smooth':
            smooth_coeff = training_params
            
            if smooth_coeff == 0:
                step_func = lambda data, labels, batch_idx: clf_std_training_step(clf, opt, data, labels, device)
            else:
                step_func = lambda data, labels, batch_idx: clf_smoothlabel_training_step(clf, opt, data, labels, smooth_coeff, device)

        elif training_type == 'mixup':
            second_train_loader, alpha = training_params
            step_func = lambda data_1, labels_1, data_2, labels_2: clf_mixup_training_step(clf, opt, data_1, labels_1, data_2, labels_2, alpha, device=device)
        
        elif training_type == 'disturblabel':
            alpha = training_params
            step_func = lambda data, labels, batch_idx: clf_disturblabel_training_step(clf, opt, data, labels, alpha, device)
    else:
        accumulation_steps = training_params
        tot_batches = af.loader_batch_counter(train_loader)
        step_func = lambda data, labels, batch_idx: clf_dp_training_step(clf, opt, data, labels, batch_idx, accumulation_steps, tot_batches, device)

    for epoch in range(1, epochs+1):        
        print('Epoch: {}/{}'.format(epoch, epochs))
        top1_test, top5_test = test_clf(clf, test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top5 Test accuracy: {}'.format(top5_test))

        if training_type == 'mixup':
            for (x_1, y_1), (x_2, y_2) in zip(train_loader, second_train_loader):
                step_func(x_1, y_1, x_2, y_2)

                del x_1, y_1, x_2, y_2

        else:
            batch_idx = 1
            for x, y in tqdm(train_loader):

                step_func(x, y, batch_idx)
                batch_idx += 1

                del x, y
            
        if clf.is_dp:
            epsilon, best_alpha = opt.privacy_engine.get_privacy_spent(1e-5)
            clf.dp_best_alphas.append(best_alpha)
            clf.dp_epsilons.append(epsilon)

            print(f"(ε = {epsilon:.2f}, δ = {1e-5}) for α = {best_alpha}")
        
        if save_func is not None:
            save_func(clf, epoch)

        sch.step()

    top1_test, top5_test = test_clf(clf, test_loader, device)
    print('End - Top1 Test accuracy: {}'.format(top1_test))
    print('End - Top5 Test accuracy: {}'.format(top5_test))