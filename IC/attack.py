from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

'''
Evaluating robustness to adversary attack using FGSM.
Reference: Disentangled Information Bottleneck
https://github.com/PanZiqiAI/disentangled-information-bottleneck/blob/master/shared_libs/utils/evaluations.py
'''

class AdvAttackEvaluator(object):
    def __init__(self, model, device, epsilon_list):
        # Modules
        self._model = model
        # Configs
        self._device = device
        self._epsilon_list = epsilon_list

    def _get_output(self, x, mode, **kwargs):
        assert mode in ['output', 'pred', 'acc']
        (mu_S, std_S), logits_S, (mu_T, std_T), logits_T = self._model(x)
        output = logits_S + logits_T
        if mode == 'output': return output
        pred = output.max(dim=1, keepdim=True)[1].squeeze()
        if mode == 'pred': return pred
        acc = pred == kwargs['label']
        return acc

    def _perturb_image_fgsm(self, image, label):
        image.requires_grad_(True)
        # 1. Calculate output & value
        output = self._get_output(image, mode='output')
        loss_value = F.cross_entropy(output, label)
        
        # 2. Calculate gradient
        self._model.zero_grad()
        loss_value.backward()
        
        # 3. Take gradient sign
        image_grad_sign = image.grad.data.sign()
        perturbed_image_list = [torch.clamp(image + eps * image_grad_sign, min=-1, max=1) for eps in self._epsilon_list]
        # Return
        return perturbed_image_list

    def __call__(self, dataloader):
        attack_acc_list = [[] for _ in self._epsilon_list]
        for batch_index, batch_data in enumerate(tqdm(dataloader)):
            image, label, _ = map(lambda _x: _x.to(self._device), batch_data)
            # 1. Get correct
            batch_acc_init = self._get_output(image, mode='acc', label=label)
            # (1) Get indices
            batch_indices = np.argwhere(batch_acc_init.cpu().numpy())[:, 0]
            batch_indices = None if len(batch_indices) == 0 else torch.LongTensor(batch_indices).to(self._device)
            if batch_indices is None: continue
            # (2) Get samples & labels
            correct_image = torch.index_select(image, dim=0, index=batch_indices)
            correct_label = torch.index_select(label, dim=0, index=batch_indices)
            # 2. Get perturbed images
            perturbed_image_list = self._perturb_image_fgsm(correct_image, correct_label)
            # 3. Re-classify perturbed image
            batch_perturbed_acc_list = [self._get_output(perturbed_image, label=correct_label, mode='acc')
                                        for perturbed_image in perturbed_image_list]
            # Save
            for eps_index, batch_perturbed_acc in enumerate(batch_perturbed_acc_list):
                attack_acc_list[eps_index].append(batch_perturbed_acc)
        attack_acc_list = [torch.cat(aa, dim=0).float().mean() for aa in attack_acc_list]
        # Return
        return {'eps_%.2f' % eps: aa for eps, aa in zip(self._epsilon_list, attack_acc_list)}



class AdvAttackEvaluator_VIB(object):
    def __init__(self, model, device, epsilon_list):
        # Modules
        self._model = model
        # Configs
        self._device = device
        self._epsilon_list = epsilon_list

    def _get_output(self, x, mode, **kwargs):
        assert mode in ['output', 'pred', 'acc']
        (mu, std), output = self._model(x)
        if mode == 'output': return output
        pred = output.max(dim=1, keepdim=True)[1].squeeze()
        if mode == 'pred': return pred
        acc = pred == kwargs['label']
        return acc

    def _perturb_image_fgsm(self, image, label):
        image.requires_grad_(True)
        # 1. Calculate output & value
        output = self._get_output(image, mode='output')
        loss_value = F.cross_entropy(output, label)
        
        # 2. Calculate gradient
        self._model.zero_grad()
        loss_value.backward()
        
        # 3. Take gradient sign
        image_grad_sign = image.grad.data.sign()
        perturbed_image_list = [torch.clamp(image + eps * image_grad_sign, min=-1, max=1) for eps in self._epsilon_list]
        # Return
        return perturbed_image_list

    def __call__(self, dataloader):
        attack_acc_list = [[] for _ in self._epsilon_list]
        for batch_index, batch_data in enumerate(tqdm(dataloader)):
            image, label, _ = map(lambda _x: _x.to(self._device), batch_data)
            # 1. Get correct
            batch_acc_init = self._get_output(image, mode='acc', label=label)
            # (1) Get indices
            batch_indices = np.argwhere(batch_acc_init.cpu().numpy())[:, 0]
            batch_indices = None if len(batch_indices) == 0 else torch.LongTensor(batch_indices).to(self._device)
            if batch_indices is None: continue
            # (2) Get samples & labels
            correct_image = torch.index_select(image, dim=0, index=batch_indices)
            correct_label = torch.index_select(label, dim=0, index=batch_indices)
            # 2. Get perturbed images
            perturbed_image_list = self._perturb_image_fgsm(correct_image, correct_label)
            # 3. Re-classify perturbed image
            batch_perturbed_acc_list = [self._get_output(perturbed_image, label=correct_label, mode='acc')
                                        for perturbed_image in perturbed_image_list]
            # Save
            for eps_index, batch_perturbed_acc in enumerate(batch_perturbed_acc_list):
                attack_acc_list[eps_index].append(batch_perturbed_acc)
        attack_acc_list = [torch.cat(aa, dim=0).float().mean() for aa in attack_acc_list]
        # Return
        return {'eps_%.2f' % eps: aa for eps, aa in zip(self._epsilon_list, attack_acc_list)}