import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from torch.autograd import Variable
from numbers import Number
from torch_geometric.utils import softmax, degree


class Mixture_Gaussian_reparam(nn.Module):
    def __init__(
        self,
        # Use as reparamerization:
        mean_list=None,
        scale_list=None,
        weight_logits=None,
        # Use as prior:
        Z_size=None,
        n_components=None,
        mean_scale=0.1,
        scale_scale=0.1,
        # Mode:
        is_reparam=True,
        reparam_mode="diag",
        is_cuda=False,
    ):
        super(Mixture_Gaussian_reparam, self).__init__()
        self.is_reparam = is_reparam
        self.reparam_mode = reparam_mode
        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")

        if self.is_reparam:
            self.mean_list = mean_list         # size: [B, Z, k]
            self.scale_list = scale_list       # size: [B, Z, k]
            self.weight_logits = weight_logits # size: [B, k]
            self.n_components = self.weight_logits.shape[-1]
            self.Z_size = self.mean_list.shape[-2]
        else:
            self.n_components = n_components
            self.Z_size = Z_size
            self.mean_list = nn.Parameter((torch.rand(1, Z_size, n_components) - 0.5) * mean_scale)
            self.scale_list = nn.Parameter(torch.log(torch.exp((torch.rand(1, Z_size, n_components) * 0.2 + 0.9) * scale_scale) - 1))
            self.weight_logits = nn.Parameter(torch.zeros(1, n_components))
            if mean_list is not None:
                self.mean_list.data = to_Variable(mean_list)
                self.scale_list.data = to_Variable(scale_list)
                self.weight_logits.data = to_Variable(weight_logits)

        self.to(self.device)


    def log_prob(self, input):
        """Obtain the log_prob of the input."""
        input = input.unsqueeze(-1)  # [S, B, Z, 1]
        if self.reparam_mode == "diag":
            if self.is_reparam:
                logits = - (input - self.mean_list) ** 2 / 2 / self.scale_list ** 2 - torch.log(self.scale_list * np.sqrt(2 * np.pi))
            else:
                scale_list = F.softplus(self.scale_list, beta=1)
                logits = - (input - self.mean_list) ** 2 / 2 / scale_list ** 2 - torch.log(scale_list * np.sqrt(2 * np.pi))
        else:
            raise
        log_prob = torch.logsumexp(logits + F.log_softmax(self.weight_logits, -1).unsqueeze(-2), axis=-1)  # F(...).unsqueeze(-2): [B, 1, k]
        return log_prob


    def prob(self, Z):
        return torch.exp(self.log_prob(Z))


    def sample(self, n=None):
        if n is None:
            n_core = 1
        else:
            assert isinstance(n, tuple)
            n_core = n[0]
        weight_probs = F.softmax(self.weight_logits, -1)  # size: [B, m]
        idx = torch.multinomial(weight_probs, n_core, replacement=True).unsqueeze(-2).expand(-1, self.mean_list.shape[-2], -1)  # multinomial result: [B, S]; result: [B, Z, S]
        mean_list  = torch.gather(self.mean_list,  dim=-1, index=idx)  # [B, Z, S]
        if self.is_reparam:
            scale_list = torch.gather(self.scale_list, dim=-1, index=idx)  # [B, Z, S]
        else:
            scale_list = F.softplus(torch.gather(self.scale_list, dim=-1, index=idx), beta=1)  # [B, Z, S]
        Z = torch.normal(mean_list, scale_list).permute(2, 0, 1)
        if n is None:
            Z = Z.squeeze(0)
        return Z


    def rsample(self, n=None):
        return self.sample(n=n)


    def __repr__(self):
        return "Mixture_Gaussian_reparam({}, Z_size={})".format(self.n_components, self.Z_size)

def to_Variable(*arrays, **kwargs):
    """Transform numpy arrays into torch tensors/Variables"""
    is_cuda = kwargs["is_cuda"] if "is_cuda" in kwargs else False
    requires_grad = kwargs["requires_grad"] if "requires_grad" in kwargs else False
    array_list = []
    for array in arrays:
        is_int = False
        if isinstance(array, Number):
            is_int = True if isinstance(array, int) else False
            array = [array]
        if isinstance(array, np.ndarray) or isinstance(array, list) or isinstance(array, tuple):
            is_int = True if np.array(array).dtype.name == "int64" else False
            array = torch.tensor(array).float()
        if isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor):
            array = Variable(array, requires_grad=requires_grad)
        if "preserve_int" in kwargs and kwargs["preserve_int"] is True and is_int:
            array = array.long()
        array = set_cuda(array, is_cuda)
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list

def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list

def record_data(data_record_dict, data_list, key_list, nolist=False, ignore_duplicate=False):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    if not isinstance(data_list, list):
        data_list = [data_list]
    if not isinstance(key_list, list):
        key_list = [key_list]
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if nolist:
            data_record_dict[key] = data
        else:
            if key not in data_record_dict:
                data_record_dict[key] = [data]
            else: 
                if (not ignore_duplicate) or (data not in data_record_dict[key]):
                    data_record_dict[key].append(data)
    
def set_cuda(tensor, is_cuda):
    if isinstance(is_cuda, str):
        return tensor.cuda(is_cuda)
    else:
        if is_cuda:
            return tensor.cuda()
        else:
            return tensor

def reparameterize(model, input, mode="full", size=None):
    if mode.startswith("diag"):
        if model is not None and model.__class__.__name__ == "Mixture_Model":
            return reparameterize_mixture_diagonal(model, input, mode=mode)
        else:
            return reparameterize_diagonal(model, input, mode=mode)
    elif mode == "full":
        return reparameterize_full(model, input, size=size)
    else:
        raise Exception("Mode {} is not valid!".format(mode))


def reparameterize_diagonal(model, input, mode):
    if model is not None:
        mean_logit = model(input)
    else:
        mean_logit = input
    if mode.startswith("diagg"):
        if isinstance(mean_logit, tuple):
            mean = mean_logit[0]
        else:
            mean = mean_logit
        std = torch.ones(mean.shape).to(mean.device)
        dist = Normal(mean, std)
        return dist, (mean, std)
    elif mode.startswith("diag"):
        if isinstance(mean_logit, tuple):
            mean_logit = mean_logit[0]
        size = int(mean_logit.size(-1) / 2)
        mean = mean_logit[:, :size]
        std = F.softplus(mean_logit[:, size:], beta=1) + 1e-10
        dist = Normal(mean, std)
        return dist, (mean, std)
    else:
        raise Exception("mode {} is not valid!".format(mode))


def reparameterize_mixture_diagonal(model, input, mode):
    mean_logit, weight_logits = model(input)
    if mode.startswith("diagg"):
        mean_list = mean_logit
        scale_list = torch.ones(mean_list.shape).to(mean_list.device)
    else:
        size = int(mean_logit.size(-2) / 2)
        mean_list = mean_logit[:, :size]
        scale_list = F.softplus(mean_logit[:, size:], beta=1) + 0.01  # Avoid the std to go to 0
    dist = Mixture_Gaussian_reparam(mean_list=mean_list,
                                    scale_list=scale_list,
                                    weight_logits=weight_logits,
                                   )
    return dist, (mean_list, scale_list)


def reparameterize_full(model, input, size=None):
    if model is not None:
        mean_logit = model(input)
    else:
        mean_logit = input
    if isinstance(mean_logit, tuple):
        mean_logit = mean_logit[0]
    if size is None:
        dim = mean_logit.size(-1)
        size = int((np.sqrt(9 + 8 * dim) - 3) / 2)
    mean = mean_logit[:, :size]
    scale_tril = fill_triangular(mean_logit[:, size:], size)
    scale_tril = matrix_diag_transform(scale_tril, F.softplus)
    dist = MultivariateNormal(mean, scale_tril = scale_tril)
    return dist, (mean, scale_tril)


def sample(dist, n=None):
    """Sample n instances from distribution dist"""
    if n is None:
        return dist.rsample()
    else:
        return dist.rsample((n,))
    
def fill_triangular(vec, dim, mode = "lower"):
    """Fill an lower or upper triangular matrices with given vectors"""
    num_examples, size = vec.shape
    assert size == dim * (dim + 1) // 2
    matrix = torch.zeros(num_examples, dim, dim).to(vec.device)
    idx = (torch.tril(torch.ones(dim, dim)) == 1).unsqueeze(0)
    idx = idx.repeat(num_examples,1,1)
    if mode == "lower":
        matrix[idx] = vec.contiguous().view(-1)
    elif mode == "upper":
        matrix[idx] = vec.contiguous().view(-1)
    else:
        raise Exception("mode {} not recognized!".format(mode))
    return matrix

def matrix_diag_transform(matrix, fun):
    """Return the matrices whose diagonal elements have been executed by the function 'fun'."""
    num_examples = len(matrix)
    idx = torch.eye(matrix.size(-1)).bool().unsqueeze(0)
    idx = idx.repeat(num_examples, 1, 1)
    new_matrix = matrix.clone()
    new_matrix[idx] = fun(matrix.diagonal(dim1 = 1, dim2 = 2).contiguous().view(-1))
    return new_matrix


def get_reparam_num_neurons(out_channels, reparam_mode):
    if reparam_mode is None or reparam_mode == "None":
        return out_channels
    elif reparam_mode == "diag":
        return out_channels * 2
    elif reparam_mode == "full":
        return int((out_channels + 3) * out_channels / 2)
    else:
        raise "reparam_mode {} is not valid!".format(reparam_mode)
    
def sample_lognormal(mean, sigma=None, sigma0=1.):
    """
    Samples from a log-normal distribution using the reparametrization
    trick so that we can backprogpagate the gradients through the sampling.
    By setting sigma0=0 we make the operation deterministic (useful at testing time)
    """
    e = torch.randn(mean.shape).to(sigma.device)
    return torch.exp(mean + sigma * sigma0 * e)

def scatter_sample(src, index, temperature, num_nodes=None):
    gumbel = torch.distributions.Gumbel(torch.tensor([0.]).to(src.device), 
            torch.tensor([1.0]).to(src.device)).sample(src.size()).squeeze(-1)
    log_prob = torch.log(src+1e-16)
    logit = (log_prob + gumbel) / temperature
    print(num_nodes, type(num_nodes))
    return softmax(logit, index, num_nodes=num_nodes)

def uniform_prior(index):
    deg = degree(index)
    deg = deg[index]
    return 1./deg.unsqueeze(1)

def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return

def discriminator_loss(output, joint=False):
    if joint:
        target = torch.ones_like(output)
    elif not joint:
        target = torch.zeros_like(output)
    return F.cross_entropy(output, target)