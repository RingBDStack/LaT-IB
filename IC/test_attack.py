import torch
import argparse
import numpy as np
from data.datasets import input_dataset
from attack import *

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
        
# Hyper Parameters
noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                  'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label',
                  'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
    
def set_global_seeds(i):
    np.random.seed(i)
    torch.manual_seed(i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(i)

set_global_seeds(args.seed)
warm_up = args.pretrain_ep
injection = args.injection_ep

train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human, args.noise_mode, args.noise_rate)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False)

LaT_IB = torch.load(f"NAME.pth.tar")
print('load ok')
attacker = AdvAttackEvaluator(LaT_IB, 'cuda', [0.05, 0.1, 0.2])
train_attack_acc = attacker(train_loader)
test_attack_acc = attacker(test_loader)
items = {'train_%s' % k: v for k, v in train_attack_acc.items()}
items.update({'test_%s' % k: v for k, v in test_attack_acc.items()})
print(items)
