import torch
import torch.nn.functional as F
from loss import *
from utils import *
from data_loader import *
from model import robust_GIB, Discriminator
from arg import get_args
import time
import math

# ==================== Argparse ====================
args = get_args()

dataset_name = args.dataset
noise_type = args.noise_type
noise_rate = args.noise_ratio
random_state = args.random_state
hidden_channels = args.hid_dim
lr = args.lr
weight_decay = args.weight_decay
warm_up = args.warmup
injection = args.injection
epoches = args.epoch
sel_ratio = args.sel_ratio
beta1 = args.beta1
beta2 = args.beta2
gamma = args.gamma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_use_mean = True
reparam_all_layers = (-1, )  # -1 means the last layer of encoder
reparam_mode = "diag"
prior_mode = "mixGau-100"
struct_dropout_mode = ("DNsampling",'Bernoulli',0.1,0.5,"norm",2)

# ==================== Data Process ====================
dataset, data = get_data(dataset_name, split=args.split, seed=random_state, hop=struct_dropout_mode[-1])
data, ori_indices, clean_indices, noise_indices = add_noise(data, dataset.num_classes, 
                                                            noise_rate=noise_rate, noise_type=noise_type, 
                                                            random_state=random_state)
data = data.to(device)

# ==================== Model Set ====================
model = robust_GIB(
    model_type='GAT',  # Only GAT can get the structure KL loss
    num_features=dataset.num_node_features,
    num_classes=dataset.num_classes,
    reparam_mode=reparam_mode,
    prior_mode=prior_mode,
    struct_dropout_mode=struct_dropout_mode,
    latent_size=hidden_channels,
    num_layers=args.layers,
    val_use_mean=val_use_mean,
    reparam_all_layers=reparam_all_layers,
    is_cuda=True,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
discriminator = Discriminator(input_dim=hidden_channels, num_classes=dataset.num_classes)
discriminator = discriminator.to(device)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=weight_decay)
loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

# ==================== Training Function ====================
def select(model, mask=None, jsd = False):
    """
    InfoJS selector
    """
    model.eval()
    targets_list = torch.zeros(len(mask), dtype=torch.long)
    with torch.no_grad():
        reg_info_S, (mu_S, std_S), out_c, reg_info_T, (mu_T, std_T), out_n = model(data)
        js_div = js_gaussian_diag(mu_S[mask], std_S[mask], mu_T[mask], std_T[mask], reduction='none').cpu()
        loss_S = F.cross_entropy(out_c[mask], data.y[mask], reduction='none').cpu()
        targets_list = data.y[mask].cpu()

    sel_S = torch.zeros_like(data.y)
    sel_T = torch.zeros_like(data.y)
    idx_chosen_S, idx_chosen_T = [], []
    # select based I(S;Y)
    for j in range(dataset.num_classes):
        indices = torch.where(targets_list == j)[0]
        if len(indices) == 0:
            continue
        
        bs_j = targets_list.shape[0] * (1. / dataset.num_classes)
        partition_j = max(min(int(math.ceil(bs_j * sel_ratio)), len(indices)), 1)

        pseudo_loss_S = loss_S[indices]

        sorted_idx_S = torch.argsort(pseudo_loss_S)[:partition_j]
        sorted_idx_T = torch.argsort(pseudo_loss_S, descending=True)[:partition_j]
        idx_chosen_S.append(indices[sorted_idx_S])
        idx_chosen_T.append(indices[sorted_idx_T])

    idx_chosen_S = torch.cat(idx_chosen_S)
    idx_chosen_T = torch.cat(idx_chosen_T)

    global_mask_idx = torch.where(mask)[0]
    global_idx_chosen_S = global_mask_idx[idx_chosen_S]
    global_idx_chosen_T = global_mask_idx[idx_chosen_T]

    sel_S[global_idx_chosen_S] = 1
    sel_T[global_idx_chosen_T] = 1

    if jsd:
        # select based JSD
        idx_chosen_S, idx_chosen_T = [], []
        for j in range(dataset.num_classes):
            indices = torch.where(targets_list == j)[0]
            if len(indices) == 0:
                continue
            
            bs_j = targets_list.shape[0] * (1. / dataset.num_classes)
            partition_j = max(min(int(math.ceil(bs_j * sel_ratio)), len(indices)), 1)
    
            js_tmp = js_div[indices]
    
            sorted_idx_S = torch.argsort(js_tmp)[:partition_j]
            sorted_idx_T = torch.argsort(js_tmp, descending=True)[:partition_j]
            idx_chosen_S.append(indices[sorted_idx_S])
            idx_chosen_T.append(indices[sorted_idx_T])
    
        idx_chosen_S = torch.cat(idx_chosen_S)
        idx_chosen_T = torch.cat(idx_chosen_T)
    
        global_mask_idx = torch.where(mask)[0]
        global_idx_chosen_S = global_mask_idx[idx_chosen_S]
        global_idx_chosen_T = global_mask_idx[idx_chosen_T]
    
        sel_S[global_idx_chosen_S] = 1
        sel_T[global_idx_chosen_T] = 1

    return sel_S, sel_T

def get_ixz_loss(reg_info_S, reg_info_T, beta1, beta2):
    loss = 0
    if beta1 is not None and beta1 != 0:
        ixz = torch.stack(reg_info_S["ixz_list"], 1).mean(0).sum()
        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            ixz = ixz + torch.stack(reg_info_S["ixz_DN_list"], 1).mean(0).sum()
        loss = loss + ixz * beta1
        ixz = torch.stack(reg_info_T["ixz_list"], 1).mean(0).sum()
        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            ixz = ixz + torch.stack(reg_info_T["ixz_DN_list"], 1).mean(0).sum()
        loss = loss + ixz * beta1
    if beta2 is not None and beta2 != 0:
        structure_kl_loss = torch.stack(reg_info_S["structure_kl_list"]).mean()
        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            structure_kl_loss = structure_kl_loss + torch.stack(reg_info_S["structure_kl_DN_list"]).mean()
        loss = loss + structure_kl_loss * beta2
        structure_kl_loss = torch.stack(reg_info_T["structure_kl_list"]).mean()
        if model.struct_dropout_mode[0] == 'DNsampling' or (model.struct_dropout_mode[0] == 'standard' and len(model.struct_dropout_mode) == 3):
            structure_kl_loss = structure_kl_loss + torch.stack(reg_info_T["structure_kl_DN_list"]).mean()
        loss = loss + structure_kl_loss * beta2
    return loss

def test(mask, test=False):
    model.eval()
    with torch.no_grad():
        _, _, out_c, _, _, out_n = model(data)
        pred_c = out_c.argmax(dim=1)
        pred_n = out_n.argmax(dim=1)
        pred_cn = (F.softmax(out_c, dim=1) + F.softmax(out_n, dim=1)).argmax(dim=1)
        correct_c = pred_c[mask] == data.y[mask]
        correct_n = pred_n[mask] == data.y[mask]
        correct_cn = pred_cn[mask] == data.y[mask]
        acc_c = int(correct_c.sum()) / int(mask.sum())
        acc_n = int(correct_n.sum()) / int(mask.sum())
        acc_cn = int(correct_cn.sum()) / int(mask.sum())
    if test:
        return f"{acc_c*100:.2f}", acc_c
    return f"{acc_c*100:.2f} / {acc_n*100:.2f}", acc_c

def train_discriminator(data, length, discriminator, optimizer_d):
    discriminator.train()
    (en_S, en_T, labels_tmp) = data
    en_S, en_T = en_S.detach().clone(), en_T.detach().clone()
    labels_tmp = labels_tmp.detach().clone()
    perm = torch.randperm(length)
    loss_d = discriminator_loss(discriminator(en_S, labels_tmp, en_T, labels_tmp), joint=True) + discriminator_loss(discriminator(en_S, labels_tmp, en_T[perm], labels_tmp[perm]), joint=False)
    optimizer_d.zero_grad()
    loss_d.backward()
    optimizer_d.step()
    return loss_d

# ==================== Start Training ====================
best_val = 0
best_acc = 0
best_epoch = -1
global_mask_idx = torch.where(data.train_mask)[0]
jsd = False

time_start = time.time()

for epoch in range(epoches):
    if epoch == warm_up:
        period1_time = time.time()
    elif epoch == warm_up + injection:
        period2_time = time.time()
    
    # Warmup stage
    if epoch < warm_up:
        model.train()
        reg_info_S, _, out_c, reg_info_T, _, out_n = model(data)
        loss = loss_func(out_c[data.train_mask], data.y[data.train_mask]) + loss_func(out_n[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Knowledge Injection stage       
    elif epoch < warm_up + injection:
        if epoch - warm_up - injection > 10:  # at first, S \sim T.
            jsd = True
        # select based InfoJS selector
        sel_S, sel_T = select(model, mask=data.train_mask, jsd=jsd)
        sel_S = sel_S.view(-1, 1).type(torch.FloatTensor)
        sel_T = sel_T.view(-1, 1).type(torch.FloatTensor)
        sel_S = sel_S.to(device)
        sel_T = sel_T.to(device)

        model.train()
        reg_info_S, (mu_S, std_S), out_c, reg_info_T, (mu_T, std_T), out_n = model(data)
        js_div = js_gaussian_diag(mu_S, std_S, mu_T, std_T, reduction='none')

        # select based predicted confidence
        labels = F.one_hot(data.y, num_classes=dataset.num_classes).float().to(device)
        with torch.no_grad():
            p_S = torch.softmax(out_c, dim=1)
            p_T = torch.softmax(out_n, dim=1)
            pred_net_S = F.one_hot(p_S.max(dim=1)[1], dataset.num_classes).float()
            pred_net_T = F.one_hot(p_T.max(dim=1)[1], dataset.num_classes).float()

            conf_scores = (labels[data.train_mask] * p_S[data.train_mask]).sum(dim=1)
            high_conf_cond = conf_scores > args.ub
            low_conf_cond = conf_scores < args.lb
            sel_S[global_mask_idx[high_conf_cond]] = 1
            sel_T[global_mask_idx[low_conf_cond]] = 1

            idx_chosen_S = torch.where(sel_S[data.train_mask] == 1)[0]
            idx_chosen_T = torch.where(sel_T[data.train_mask] == 1)[0]
            idx_unchosen = torch.where((sel_S[data.train_mask] != 1) & (sel_T[data.train_mask] != 1))[0]
            idx_chosen_S = global_mask_idx[idx_chosen_S]
            idx_chosen_T = global_mask_idx[idx_chosen_T]
            idx_unchosen = global_mask_idx[idx_unchosen]

        loss_S = loss_func(out_c[idx_chosen_S], data.y[idx_chosen_S]) - js_div[idx_chosen_S].mean() * args.beta
        loss_T = loss_func(out_n[idx_chosen_T], data.y[idx_chosen_T]) - js_div[idx_chosen_T].mean() * args.beta

        loss = loss_S + loss_T + get_ixz_loss(reg_info_S, reg_info_T, beta1, beta2)

        if len(idx_unchosen) > 0:
            loss_both = loss_func(out_c[idx_unchosen], pred_net_S[idx_unchosen]) + js_div[idx_unchosen].mean() * args.beta + \
                        loss_func(out_n[idx_unchosen], pred_net_T[idx_unchosen]) + js_div[idx_unchosen].mean() * args.beta
            loss += loss_both
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Robust Training stage
    else:
        model.train()
        reg_info_S, _, out_c, reg_info_T, _, out_n = model(data)
        loss = ConCE(out_c[data.train_mask], out_n[data.train_mask], data.y[data.train_mask], thod=args.thod, noise_mode=args.noise_type)
        # Add IB loss:
        loss += get_ixz_loss(reg_info_S, reg_info_T, beta1, beta2)
        # I(S,Y;T,Y)
        en_S, _, _ = model.encoder_S(data)
        en_T, _, _ = model.encoder_T(data)
        labels_tmp = torch.zeros(data.x.size(0), dataset.num_classes, device=device).scatter_(1, data.y.view(-1, 1), 1)
        loss_joint = discriminator_loss(discriminator(en_S[data.train_mask], labels_tmp[data.train_mask], en_T[data.train_mask], labels_tmp[data.train_mask]), joint=True)
        loss += gamma * loss_joint
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train discriminator
        model.eval()
        loss_dis = train_discriminator((en_S[data.train_mask], en_T[data.train_mask], labels_tmp[data.train_mask]), data.train_mask.sum(), discriminator, optimizer_d)

    # Logging
    train_log, train_acc = test(data.train_mask)
    val_log, val_acc = test(data.val_mask)
    test_log, test_acc = test(data.test_mask, test=True)
    if best_val < val_acc:
        best_val = val_acc
        best_acc = test_acc
        best_epoch = epoch
        # torch.save(model.state_dict(), f'MODEL_NAME')
    if epoch % args.pre_print == 0 or epoch == epoches - 1:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f};Train Accuracy: {train_log}, Valid Accuracy: {val_log}, Test Accuracy: {test_log}%")

print(f"Training complete! Best_acc = {best_acc * 100:.2f}%, Last_acc = {test_acc * 100:.2f}%, Best_epoch = {best_epoch}")

# Time consumption
time_end = time.time()
print("Training Time: " + str(time_end - time_start) + "s")
stage1 = round((period1_time - time_start) / warm_up, 2)
stage2 = round((period2_time - period1_time) / injection, 2)
stage3 = round((time_end - period2_time) / (100 - warm_up - injection), 2)
print(f"Warmup: {stage1} s")
print(f"Knowledge Injection: {stage2} s")
print(f"Robust Training: {stage3} s")

