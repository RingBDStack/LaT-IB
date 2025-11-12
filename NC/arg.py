import argparse

def print_args(args):
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

def get_args(cfg = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset used.')
    parser.add_argument('--split', type=str, default='per_class', help='for dblp, sampler per class or percent [per_class, percent]')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup epochs.')
    parser.add_argument('--injection', type=int, default=50, help='Number of injection epochs.')
    parser.add_argument('--hid_dim', type=int, default=16, help='Dimension of hidden layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for regularization.')
    parser.add_argument('--noise_type', type=str, default='uniform', help='Type of noise applied to data.')
    parser.add_argument('--noise_ratio', type=float, default=0.1, help='Ratio of noise applied to data.')
    parser.add_argument('--random_state', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for the optimizer.')
    parser.add_argument('--sel_ratio', type=float, default=0.3, help='selection ratio.')
    parser.add_argument('--ub', default=0.95, type=float, help='high-confidence selection threshold')
    parser.add_argument('--lb', default=0.7, type=float, help='low-confidence selection threshold')
    parser.add_argument('--thod', default=0.9, type=float, help='period3 loss threshold')
    parser.add_argument('--beta1', default=0.001, type=float, help='for I(D;S,T) feather loss')
    parser.add_argument('--beta2', default=0.01, type=float, help='for I(D;S,T) structure kl loss')
    parser.add_argument('--beta', default=0.001, type=float, help='for js div loss')
    parser.add_argument('--gamma', default=0.01, type=float, help='for I(S;T|Y)')
    parser.add_argument('--layers', default=2, type=int, help='')
    parser.add_argument('--pre_print', default=10, type=int, help='')

    args = parser.parse_args()

    print_args(args)

    return args

def get_cfg(args):
    # 读取config文件并更新args参数


    return args
