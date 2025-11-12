import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
from model.modelIB import *
from utils.fmix import *
# from utils.mix import *
from utils.loss import *
import time
from data.datasets import input_dataset
import copy
from tqdm import tqdm
from attack import *
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--sel_ratio', default=0.5, type=float, help='ratio of selecting')
parser.add_argument('--ub', default=0.95, type=float, help='high-confidence selection threshold')
parser.add_argument('--lb', default=0.5, type=float, help='low-confidence selection threshold')
parser.add_argument('--thod', default=0.85, type=float, help='ConCE loss threshold')
parser.add_argument('--beta', default=0.001, type=float, help='beta for I(D;S,T)')
parser.add_argument('--gamma', default=0.01, type=float, help='gamma for I(S;T|Y)')

parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--pretrain_ep', default=20, type=int, help='Warmup epoch')
parser.add_argument('--injection_ep', default=80, type=int, help='Knowledge Injection epoch')

parser.add_argument('--load', action='store_true', default=False, help='load model')

parser.add_argument('--debias_output', default=0.8, type=float, help='debias strength for loss calculation')
parser.add_argument('--debias_pl', default=0.8, type=float, help='debias strength for pseudo-label generation')
parser.add_argument('--bias_m', default=0.9999, type=float, help='moving average parameter of bias estimation')

parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noise_mode', default='cifarn', type=str,help='cifarn, sym, asym')
parser.add_argument('--noise_type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_rate', default=0.2, type=float, help='noise rate for synthetic noise')
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--noise_path', type=str, help='path of CIFAR-10/100_human.pt', default=None)

parser.add_argument('--z_dim', type=int, default=128, help='hidden dim')
parser.add_argument('--EMA', type=float, default=0.8, help='')
args = parser.parse_args()
print(args)

sel_ratio = args.sel_ratio
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'

noise_type = copy.deepcopy(args.noise_type)
# Hyper Parameters
noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                  'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label',
                  'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]

##################################### warm up ################################################
def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (image, labels, _) in enumerate(dataloader):
        image, labels = image.cuda(), labels.cuda()
        optimizer.zero_grad()
        (_, _), out_c, (_, _), out_n = net(image)
        l_ce_S = F.cross_entropy(out_c, labels)
        l_ce_T = F.cross_entropy(out_n, labels)
        loss = l_ce_S + l_ce_T
        loss.backward()
        optimizer.step()
        if batch_idx % args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t Loss: %.4f'  % (epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))

##################################### knowledge Injection ################################################
def knowledge_injection(epoch, net, optimizer, data_loader, sel_S_all, sel_T_all, pi_S, pi_T):
    epsilon = 1e-6
    net.train()
    w = linear_rampup(epoch, args.num_epochs)
    num_iter = (len(data_loader.dataset) // args.batch_size) + 1
    
    for batch_idx, (image, label, index) in enumerate(data_loader):
        batch_size = image.size(0)
        sel_S, sel_T = sel_S_all[index], sel_T_all[index]
        sel_S = sel_S.view(-1, 1).type(torch.FloatTensor)
        sel_T = sel_T.view(-1, 1).type(torch.FloatTensor)
        label = torch.zeros(batch_size, num_classes).scatter_(1, label.view(-1, 1), 1)
        image, label, sel_S , sel_T= image.cuda(), label.cuda(), sel_S.cuda(), sel_T.cuda()
        
        (mu_S, std_S), out_c, (mu_T, std_T), out_n = net(image)
        js_div = js_gaussian_diag(mu_S, std_S, mu_T, std_T, reduction='none')
        out_c_copy = out_c.clone().detach()
        out_n_copy = out_n.clone().detach()
        out_c = debias_output(out_c, pi_S)
        out_n = debias_output(out_n, pi_T)
        
        with torch.no_grad():
            p_S = torch.softmax(out_c_copy, dim=1)
            p_T = torch.softmax(out_n_copy, dim=1)
            
            debias_p_S = debias_pl(out_c_copy, pi_S)
            debias_p_T = debias_pl(out_n_copy, pi_T)
            
            conf_scores = (label * p_S).sum(dim=1)
            high_conf_cond = conf_scores > args.ub
            low_conf_cond = conf_scores < args.lb
            sel_S[high_conf_cond] = 1
            sel_T[low_conf_cond] = 1

            idx_chosen_S = torch.where(sel_S == 1)[0]
            idx_chosen_T = torch.where(sel_T == 1)[0]
            idx_unchosen = torch.where((sel_S != 1) & (sel_T != 1))[0]

        # data argumentation for S or T
        X_S_agu = image[idx_chosen_S]
        label_agu_S = label[idx_chosen_S]
        x_fmix_S = fmix(X_S_agu)
        (_, _), logits_fmix_S, (_, _), _ = net(x_fmix_S)
        logits_fmix_S = debias_output(logits_fmix_S, pi_S)
        loss_fmix_S = fmix.loss(logits_fmix_S, label_agu_S.long())
        loss_ce_S = F.cross_entropy(out_c[idx_chosen_S], label[idx_chosen_S])
        loss_net_S = loss_ce_S + w * loss_fmix_S - js_div[idx_chosen_S].mean() * 1e-4

        X_T_agu = image[idx_chosen_T]
        label_agu_T = label[idx_chosen_T]
        x_fmix_T = fmix(X_T_agu)
        (_, _), _, (_, _), logits_fmix_T = net(x_fmix_T)
        logits_fmix_T = debias_output(logits_fmix_T, pi_T)
        loss_fmix_T = fmix.loss(logits_fmix_T, label_agu_T.long())
        loss_ce_T = F.cross_entropy(out_n[idx_chosen_T], label[idx_chosen_T])
        loss_net_T = loss_ce_T + w * loss_fmix_T - js_div[idx_chosen_T].mean() * 1e-4

        loss = loss_net_S + loss_net_T

        # both parts of S and T
        if len(idx_unchosen) > 0:
            loss_both = F.cross_entropy(out_c[idx_unchosen], debias_p_S[idx_unchosen]) + js_div[idx_unchosen].mean() * 1e-4 + \
                        F.cross_entropy(out_n[idx_unchosen], debias_p_T[idx_unchosen]) + js_div[idx_unchosen].mean() * 1e-4
            loss += loss_both

        info_loss_S = -0.5 * (1 + 2 * (std_S + epsilon).log() - mu_S.pow(2) - std_S.pow(2)).sum(1).mean().div(math.log(2))
        info_loss_T = -0.5 * (1 + 2 * (std_T + epsilon).log() - mu_T.pow(2) - std_T.pow(2)).sum(1).mean().div(math.log(2))
        loss += args.beta * (info_loss_S + info_loss_T)
            
        pi_S = bias_update(p_S[idx_chosen_S], pi_S)
        pi_T = bias_update(p_T[idx_chosen_T], pi_T)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_freq == 0 :
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t Net_S loss: %.2f  Net_T loss: %.2f'
                         % (epoch, args.num_epochs, batch_idx + 1, num_iter, loss_net_S.item(), loss_net_T.item()))

    return pi_S, pi_T

##################################### Robust Training ################################################
def train_main(epoch, model, optimizer, data_loader, discriminator, optimizer_d, all_labels = None): 
    epsilon = 1e-6
    num_iter = (len(data_loader.dataset) // args.batch_size) + 1
    model.train()
    for batch_idx, (images, labels, indexes) in enumerate(data_loader):
        # train model
        discriminator.eval()
        model.train()

        images = images.to(device)
        if all_labels == None:
            labels = labels.to(device)
            labels_tmp = torch.zeros(images.size(0), num_classes, device=device).scatter_(1, labels.view(-1, 1), 1)
        else:
            labels = all_labels[indexes].to(device)
            labels_tmp = torch.zeros(images.size(0), num_classes, device=device).scatter_(1, labels.view(-1, 1), 1)
       
        (mu_S, std_S), out_c, (mu_T, std_T), out_n = model(images)
        (_, _), en_S = model.S(images)
        (_, _), en_T = model.T(images)
        loss_joint = discriminator_loss(discriminator(en_S, labels_tmp, en_T, labels_tmp), joint=True)

        class_loss = ConCE(out_c, out_n, labels, thod=args.thod, n_type=args.noise_mode)
        info_loss_S = -0.5 * (1 + 2 * (std_S + epsilon).log() - mu_S.pow(2) - std_S.pow(2)).sum(1).mean().div(math.log(2))
        info_loss_T = -0.5 * (1 + 2 * (std_T + epsilon).log() - mu_T.pow(2) - std_T.pow(2)).sum(1).mean().div(math.log(2))
        loss = class_loss + args.beta * (info_loss_S + info_loss_T) + args.gamma * loss_joint

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train discriminator
        model.eval()
        loss_dis = train_discriminator((en_S, en_T, labels_tmp), len(indexes), discriminator, optimizer_d)
        
        if batch_idx % args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t Class loss: %.2f  Info loss: %.2f, %.2f | Discriminator loss: %.5f'
                         % (epoch, args.num_epochs, batch_idx + 1, num_iter, class_loss.item(), info_loss_S.item(), info_loss_T.item(), loss_dis.item()))


def train_discriminator(data, length, discriminator, optimizer_d):
    discriminator.train()
    (en_S, en_T, labels_tmp) = data
    en_S, en_T = en_S.detach(), en_T.detach()
    labels_tmp = labels_tmp.detach()
    perm = torch.randperm(length)
    loss_d = discriminator_loss(discriminator(en_S, labels_tmp, en_T, labels_tmp), joint=True) + discriminator_loss(discriminator(en_S, labels_tmp, en_T[perm], labels_tmp[perm]), joint=False)
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()

    return loss_d

##################################### eval && InfoJS ################################################
def get_label(label_model=None):
    print('get label start')
    label_model.eval()
    all_preds = torch.zeros((len(train_loader.dataset), num_classes), dtype=torch.float)
    # all_preds = torch.zeros(len(train_loader.dataset), dtype=torch.long)
    with torch.no_grad():
        for i, (images, labels, indexes) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            (mu_S, std_S), out_c, (mu_T, std_T), out_n = label_model(images)
            prob_s = F.softmax(out_c, dim=1)
            prob_t = F.softmax(out_n, dim=1)
            prob = (prob_s + prob_t) / 2
            all_preds[indexes.cpu()] = prob.cpu()
        return all_preds
    
def evaluate_IB(epoch, model, test_loader):
    model.eval()
    correct_S, correct_T, correct_ST = 0, 0, 0
    total = 0
    with torch.no_grad():
        for batch_idx, (image, labels, _) in enumerate(test_loader):
            image = image.cuda()
            (mu_S, std_S), out_c, (mu_T, std_T), out_n = model(image)
            outputs_S = F.softmax(out_c, dim=1)
            outputs_T = F.softmax(out_n, dim=1)
            outputs_ST = outputs_S + outputs_T
            _, pred_S = torch.max(outputs_S.data, 1)
            _, pred_T = torch.max(outputs_T.data, 1)
            _, pred_ST = torch.max(outputs_ST.data, 1)
            total += labels.size(0)
            correct_S += (pred_S.cpu() == labels).sum()
            correct_T += (pred_T.cpu() == labels).sum()
            correct_ST += (pred_ST.cpu() == labels).sum()

    # acc_ST is slightly better than acc_S
    acc_ST = 100 * float(correct_ST) / float(total)
    print("| Test Epoch #%d\t Acc: %.2f%%" % (epoch, acc_ST))

def selection(model, data_loader, sel_ratio, num_class):
    """
    InfoJS selector
    """
    model.eval()
    losses_S = torch.zeros(len(data_loader.dataset))
    losses_T = torch.zeros(len(data_loader.dataset))
    targets_list = torch.zeros(len(data_loader.dataset), dtype=torch.long)
    js_list = torch.zeros(len(data_loader.dataset), dtype=torch.float)

    with torch.no_grad():
        for images, labels, indexes in data_loader:
            images, labels = images.cuda(), labels.cuda()
            labels_cpu = labels.cpu()
            (mu_S, std_S), out_c, (mu_T, std_T), out_n = model(images)

            js_div = js_gaussian_diag(mu_S, std_S, mu_T, std_T, reduction='none').cpu()
            js_list[indexes] = js_div
            
            loss_S = F.cross_entropy(out_c, labels, reduction='none').cpu()
            loss_T = F.cross_entropy(out_n, labels, reduction='none').cpu()
            losses_S[indexes] = loss_S
            losses_T[indexes] = loss_T
            targets_list[indexes] = labels_cpu

    losses_S = (losses_S - losses_S.min()) / (losses_S.max() - losses_S.min())
    losses_T = (losses_T - losses_T.min()) / (losses_T.max() - losses_T.min())

    sel_S = torch.zeros_like(targets_list)
    sel_T = torch.zeros_like(targets_list)
    idx_chosen_S, idx_chosen_T = [], []

    for j in range(num_class):
        indices = torch.where(targets_list == j)[0]
        if len(indices) == 0:
            continue
        bs_j = targets_list.shape[0] * (1. / num_class)
        partition_j = max(min(int(math.ceil(bs_j * sel_ratio)), len(indices)), 1)
        # select based I(S;Y)
        pseudo_loss_S = losses_S[indices]
        sorted_idx_S = torch.argsort(pseudo_loss_S)[:partition_j]
        sorted_idx_T = torch.argsort(pseudo_loss_S, descending=True)[:partition_j]
        idx_chosen_S.append(indices[sorted_idx_S])
        idx_chosen_T.append(indices[sorted_idx_T])
        # select based JSD
        js_sel = js_list[indices]
        sorted_idx_S = torch.argsort(js_sel)[:partition_j]
        sorted_idx_T = torch.argsort(js_sel, descending=True)[:partition_j]
        idx_chosen_S.append(indices[sorted_idx_S])
        idx_chosen_T.append(indices[sorted_idx_T])

    idx_chosen_S = torch.cat(idx_chosen_S)
    idx_chosen_T = torch.cat(idx_chosen_T)
    sel_S[idx_chosen_S] = 1
    sel_T[idx_chosen_T] = 1

    return sel_S, sel_T

##################################### else ################################################
def create_model(model_name='resnet34'):
    model = Robust_IB(z_dim=args.z_dim, num_classes=num_classes, model_name=model_name)
    model = model.cuda()
    return model

def set_global_seeds(i):
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)

set_global_seeds(args.seed)
warm_up = args.pretrain_ep
injection = args.injection_ep


##################################### main ################################################
train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human, args.noise_mode, args.noise_rate)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False)

print(f'| Building net | numclass={num_classes}')
LaT_VIB = create_model()
if args.load:
    LaT_VIB = torch.load(f"MODEL NAME")
    print('load ok')
    evaluate_IB(-1, LaT_VIB, test_loader)

cudnn.benchmark = True

optimizer = optim.SGD(LaT_VIB.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
discriminator = Discriminator(input_dim=args.z_dim, num_classes=num_classes)
discriminator = discriminator.cuda()
optimizer_d = optim.SGD(discriminator.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

fmix = FMix()
if args.dataset == 'animal-10N':
    fmix = FMix(size=(64, 64))

pi_S = bias_initial(num_classes)
pi_T = bias_initial(num_classes)

last_label = None

start_time = time.time()

for epoch in range(args.num_epochs):
    if epoch == warm_up and epoch < warm_up + injection:
        LaT_VIB.T.reset()
        print('reset ok!')
        period1_time = time.time()
    elif epoch == warm_up + injection:
        optimizer = optim.SGD(LaT_VIB.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.num_epochs - warm_up - injection), eta_min=1e-6)
        print('set optimizer/scheduler OK')
        period2_time = time.time()

    # label update
    if epoch >= warm_up + injection and (epoch - warm_up - injection) % 10 == 0:
        if last_label == None:
            label_y = torch.tensor(train_dataset.train_noisy_labels)
            label_y_tensor = label_y.clone().detach().long()
            last_label = torch.zeros(num_training_samples, num_classes).scatter_(1, label_y_tensor.view(-1, 1), 1)
        all_label_ori = get_label(LaT_VIB)        
        last_label = args.EMA * last_label + (1 - args.EMA) * all_label_ori.detach().clone()
        _, all_labels = torch.max(last_label, 1)
        print('diff:', (all_labels != label_y_tensor).sum().item())
        print('update label ok')
    
    if epoch < warm_up:
        print('Warmup')
        warmup(epoch, LaT_VIB, optimizer, train_loader)
    elif epoch < warm_up + injection:
        print('Knowledge Injection')
        sel_S, sel_T = selection(LaT_VIB, train_loader, sel_ratio, num_classes)
        pi_S, pi_T = knowledge_injection(epoch,LaT_VIB, optimizer, train_loader, sel_S, sel_T, pi_S, pi_T)
    else:
        print('Robust Training')
        train_main(epoch, LaT_VIB, optimizer, train_loader, discriminator, optimizer_d, all_labels)

    # save model
    if args.noise_mode == 'cifarn':
        torch.save(LaT_VIB, f"./{args.dataset}_{args.noise_type}best.pth.tar")
    else:
        torch.save(LaT_VIB, f"./{args.dataset}_{args.noise_mode}_{args.noise_rate}best.pth.tar")

    scheduler.step()
    evaluate_IB(epoch, LaT_VIB, test_loader)

end_time = time.time()

# Time Consumption
print("Training Time: " + str(end_time - start_time) + "s")
stage1 = round((period1_time - start_time) / warm_up, 2)
stage2 = round((period2_time - period1_time) / injection, 2)
stage3 = round((end_time - period2_time) / (args.num_epochs - warm_up - injection), 2)
print(f"Warmup: {stage1} s")
print(f"Knowledge Injection: {stage2} s")
print(f"Robust Training: {stage3} s")

# FGSM Attack
attacker = AdvAttackEvaluator(LaT_VIB, 'cuda', [0.05, 0.1, 0.2])
train_attack_acc = attacker(train_loader)
test_attack_acc = attacker(test_loader)
items = {'train_%s' % k: v for k, v in train_attack_acc.items()}
items.update({'test_%s' % k: v for k, v in test_attack_acc.items()})
print(items)