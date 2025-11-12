from collections import OrderedDict
import itertools
import sklearn
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv
from utils import *

class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        struct_dropout_mode (tuple, optional): Choose from: None, ("standard", prob), ("info", ${MODE}),
            where ${MODE} chooses from "subset", "lognormal", "loguniform".
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, reparam_mode=None, prior_mode=None,
                 struct_dropout_mode=None, sample_size=1,
                 val_use_mean=True,
                 bias=True,
                 **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.reparam_mode = reparam_mode if reparam_mode != "None" else None
        self.prior_mode = prior_mode
        self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.struct_dropout_mode = struct_dropout_mode
        self.sample_size = sample_size
        self.val_use_mean = val_use_mean

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * self.out_neurons))
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_neurons))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.out_neurons))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(self.out_neurons))
        else:
            self.register_parameter('bias', None)
            
        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False, Z_size=self.out_channels, n_components=n_components)

        self.skip_editing_edge_index = struct_dropout_mode[0] == 'DNsampling'
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x) and not self.skip_editing_edge_index:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        out = self.propagate(edge_index, size=size, x=x)
        mean = 0
        std = 1

        if self.reparam_mode is not None:
            # Reparameterize:
            out = out.view(-1, self.out_neurons)
            self.dist, (mean, std) = reparameterize(model=None, input=out,
                                          mode=self.reparam_mode,
                                          size=self.out_channels,
                                         )  # dist: [B * head, Z]
            
            Z_core = sample(self.dist, self.sample_size)  # [S, B * head, Z]
            Z = Z_core.view(self.sample_size, -1, self.heads * self.out_channels)  # [S, B, head * Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(loc=torch.zeros(out.size(0), self.out_channels).to(x.device),
                                            scale=torch.ones(out.size(0), self.out_channels).to(x.device),
                                           )  # feature_prior: [B * head, Z]

            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = torch.distributions.kl.kl_divergence(self.dist, self.feature_prior).sum(-1).view(-1, self.heads).mean(-1)
            else:
                Z_logit = self.dist.log_prob(Z_core).sum(-1) if self.reparam_mode.startswith("diag") else self.dist.log_prob(Z_core)  # [S, B * head]
                prior_logit = self.feature_prior.log_prob(Z_core).sum(-1)  # [S, B * head]
                # upper bound of I(X; Z):
                ixz = (Z_logit - prior_logit).mean(0).view(-1, self.heads).mean(-1)  # [B]

            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)
            else:
                out = out[:, :self.out_channels].contiguous().view(-1, self.heads * self.out_channels)
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)

        if "Nsampling" in self.struct_dropout_mode[0]:
            if 'categorical' in self.struct_dropout_mode[1]:
                structure_kl_loss = torch.sum(self.alpha*torch.log((self.alpha+1e-16)/self.prior))
            elif 'Bernoulli' in self.struct_dropout_mode[1]:
                posterior = torch.distributions.bernoulli.Bernoulli(self.alpha)
                prior = torch.distributions.bernoulli.Bernoulli(self.prior) 
                structure_kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
            else:
                raise Exception("I think this belongs to the diff subset sampling that is not implemented")
        else:
            structure_kl_loss = torch.zeros([]).to(x.device)

        return out, ixz, structure_kl_loss, (mean, std)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_neurons)  # [N_edge, heads, out_channels]
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_neurons:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_neurons)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # [N_edge, heads]

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Sample attention coefficients stochastically.
        if self.struct_dropout_mode[0] == "None":
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        else:
            if self.struct_dropout_mode[0] == "standard":
                alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
                prob_dropout = self.struct_dropout_mode[1]
                alpha = F.dropout(alpha, p=prob_dropout, training=self.training)
            elif self.struct_dropout_mode[0] == "identity":
                alpha = torch.ones_like(alpha)
                alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            elif self.struct_dropout_mode[0] == "info":
                mode = self.struct_dropout_mode[1]
                if mode == "lognormal":
                    max_alpha = self.struct_dropout_mode[2] if len(self.struct_dropout_mode) > 2 else 0.7
                    alpha = 0.001 + max_alpha * alpha
                    self.kl = -torch.log(alpha/(max_alpha + 0.001))
                    sigma0 = 1. if self.training else 0.
                    alpha = sample_lognormal(mean=torch.zeros_like(alpha), sigma=alpha, sigma0=sigma0)
                else:
                    raise Exception("Mode {} for the InfoDropout is invalid!".format(mode))
            elif "Nsampling" in self.struct_dropout_mode[0]:
                neighbor_sampling_mode = self.struct_dropout_mode[1]
                if 'categorical' in neighbor_sampling_mode:
                    alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
                    self.alpha = alpha
                    self.prior = uniform_prior(edge_index_i)
                    if self.val_use_mean is False or self.training:
                        temperature = self.struct_dropout_mode[2]
                        sample_neighbor_size = self.struct_dropout_mode[3]
                        if neighbor_sampling_mode == 'categorical':
                            alpha = scatter_sample(alpha, edge_index_i, temperature, size_i)
                        elif 'multi-categorical' in neighbor_sampling_mode:
                            alphas = []
                            for _ in range(sample_neighbor_size): #! this can be improved by parallel sampling
                                alphas.append(scatter_sample(alpha, edge_index_i, temperature, size_i))
                            alphas = torch.stack(alphas, dim=0)
                            if 'sum' in neighbor_sampling_mode:
                                alpha = alphas.sum(dim=0)
                            elif 'max' in neighbor_sampling_mode:
                                alpha, _ = torch.max(alphas, dim=0)
                            else:
                                raise
                        else:
                            raise
                elif neighbor_sampling_mode == 'Bernoulli':
                    if self.struct_dropout_mode[4] == 'norm':
                        alpha_normalization = torch.ones_like(alpha)
                        alpha_normalization = softmax(alpha_normalization, edge_index_i, num_nodes=size_i)
                    alpha = torch.clamp(torch.sigmoid(alpha), 0.01, 0.99)
                    self.alpha = alpha
                    self.prior = (torch.ones_like(self.alpha)*self.struct_dropout_mode[3]).to(alpha.device)
                    if not self.val_use_mean or self.training:
                        temperature = self.struct_dropout_mode[2]
                        alpha = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.Tensor([temperature]).to(alpha.device),
                            probs=alpha).rsample()
                    if self.struct_dropout_mode[4] == 'norm':
                        alpha = alpha*alpha_normalization
                else:
                    raise
            else:
                raise
                
        return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads * self.out_neurons)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_neurons)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out


    def to_device(self, device):
        self.to(device)
        return self

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)



class GNN(torch.nn.Module):
    def __init__(
        self,
        model_type,
        num_features,
        num_classes,
        reparam_mode,
        prior_mode,
        latent_size,
        sample_size=1,
        num_layers=2,
        struct_dropout_mode=("standard", 0.6),
        dropout=True,
        with_relu=True,
        val_use_mean=True,
        reparam_all_layers=True,
        normalize=True,
        is_cuda=False,
    ):
        """Class implementing a general GNN, which can realize GAT, GIB-GAT, GCN.
        
        Args:
            model_type:   name of the base model. Choose from "GAT", "GCN".
            num_features: number of features of the data.x.
            num_classes:  number of classes for data.y.
            reparam_mode: reparameterization mode for XIB. Choose from "diag" and "full". Default "diag" that parameterizes the mean and diagonal element of the Gaussian
            prior_mode:   distribution type for the prior. Choose from "Gaussian" or "mixGau-{Number}", where {Number} is the number of components for mixture of Gaussian.
            latent_size:  latent size for each layer of GNN. If model_type="GAT", the true latent size is int(latent_size/2)
            sample_size=1:how many Z to sample for each feature X.
            num_layers=2: number of layers for the GNN
            struct_dropout_mode: Mode for how the structural representation is generated. Only effective for model_type=="GAT"
                          Choose from ("Nsampling", 'multi-categorical-sum', 0.1, 3) (here 0.1 is temperature, k=3 is the number of sampled edges with replacement), 
                          ("DNsampling", 'multi-categorical-sum', 0.1, 3, 2) (similar as above, with the local dependence range T=2) 
                          ("standard", 0.6) (standard dropout used on the attention weights in GAT)
            dropout:      whether to use dropout on features.
            with_relu:    whether to use nonlinearity for GCN.
            val_use_mean: Whether during evaluation use the parameter value instead of sampling. If True, during evaluation,
                          XIB will use mean for prediction, and AIB will use the parameter of the categorical distribution for prediction.
            reparam_all_layers: Which layers to use XIB, e.g. (1,2,4). Default (-2,), meaning the second last layer. If True, use XIB for all layers.
            normalize:    whether to normalize for GCN (only effective for GCN)
            is_cuda:      whether to use CUDA, and if so, which GPU to use. Choose from False, True, "CUDA:{GPU_ID}", where {GPU_ID} is the ID for the CUDA.
        """
        super(GNN, self).__init__()
        self.model_type = model_type
        self.num_features = num_features
        self.num_classes = num_classes
        self.normalize = normalize
        self.reparam_mode = reparam_mode
        self.prior_mode = prior_mode
        self.struct_dropout_mode = struct_dropout_mode
        self.dropout = dropout
        self.latent_size = latent_size
        self.sample_size = sample_size
        self.num_layers = num_layers
        self.with_relu = with_relu
        self.val_use_mean = val_use_mean
        self.reparam_all_layers = reparam_all_layers
        self.is_cuda = is_cuda
        self.device = torch.device(self.is_cuda if isinstance(self.is_cuda, str) else "cuda" if self.is_cuda else "cpu")

        self.init()


    def init(self):
        """Initialize the layers for the GNN."""
        self.reparam_layers = []
        if self.model_type == "GCN":
            for i in range(self.num_layers):
                if self.reparam_all_layers is True:
                    is_reparam = True
                elif isinstance(self.reparam_all_layers, tuple):
                    reparam_all_layers = tuple([kk + self.num_layers if kk < 0 else kk for kk in self.reparam_all_layers])
                    is_reparam = i in reparam_all_layers
                else:
                    raise
                if is_reparam:
                    self.reparam_layers.append(i)
                setattr(self, "conv{}".format(i + 1),
                        GCNConv(self.num_features if i == 0 else self.latent_size,
                                self.latent_size if i != self.num_layers - 1 else self.num_classes,
                                cached=True,
                                reparam_mode=self.reparam_mode if is_reparam else None,
                                prior_mode=self.prior_mode if is_reparam else None,
                                sample_size=self.sample_size,
                                bias=True if self.with_relu else False,
                                val_use_mean=self.val_use_mean,
                                normalize=self.normalize,
                ))
                
        elif self.model_type == "GAT":
            latent_size = int(self.latent_size / 2)  # Under the default setting, latent_size = 8
            for i in range(self.num_layers):
                if i == 0:
                    input_size = self.num_features
                else:
                    if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                        input_size = latent_size * 8 * 2
                    else:
                        input_size = latent_size * 8
                if self.reparam_all_layers is True:
                    is_reparam = True
                elif isinstance(self.reparam_all_layers, tuple):
                    reparam_all_layers = tuple([kk + self.num_layers if kk < 0 else kk for kk in self.reparam_all_layers])
                    is_reparam = i in reparam_all_layers
                else:
                    raise
                if is_reparam:
                    self.reparam_layers.append(i)
                setattr(self, "conv{}".format(i + 1), GATConv(
                    input_size,
                    latent_size if i != self.num_layers - 1 else self.num_classes,
                    heads=8 if i != self.num_layers - 1 else 1, concat=True,
                    reparam_mode=self.reparam_mode if is_reparam else None,
                    prior_mode=self.prior_mode if is_reparam else None,
                    val_use_mean=self.val_use_mean,
                    struct_dropout_mode=self.struct_dropout_mode,
                    sample_size=self.sample_size,
                ))
                if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                    setattr(self, "conv{}_1".format(i + 1), GATConv(
                        input_size,
                        latent_size if i != self.num_layers - 1 else self.num_classes,
                        heads=8 if i != self.num_layers - 1 else 1, concat=True,
                        reparam_mode=self.reparam_mode if is_reparam else None,
                        prior_mode=self.prior_mode if is_reparam  else None,
                        val_use_mean=self.val_use_mean,
                        struct_dropout_mode=self.struct_dropout_mode,
                        sample_size=self.sample_size,
                    ))
            # On the Pubmed dataset, use heads=8 in conv2.
        
        else:
            raise Exception("Model_type {} is not valid!".format(self.model_type))

        self.reparam_layers = sorted(self.reparam_layers)
   
        if self.model_type == "GCN":
            if self.with_relu:
                reg_params = [getattr(self, "conv{}".format(i+1)).parameters() for i in range(self.num_layers - 1)]
                self.reg_params = itertools.chain(*reg_params)
                self.non_reg_params = getattr(self, "conv{}".format(self.num_layers)).parameters()
            else:
                self.reg_params = OrderedDict()
                self.non_reg_params = self.parameters()
        else:
            self.reg_params = self.parameters()
            self.non_reg_params = OrderedDict()
        self.to(self.device)


    def set_cache(self, cached):
        """Set cache for GCN."""
        for i in range(self.num_layers):
            if hasattr(getattr(self, "conv{}".format(i+1)), "set_cache"):
                getattr(self, "conv{}".format(i+1)).set_cache(cached)


    def to_device(self, device):
        """Send all the layers to the specified device."""
        for i in range(self.num_layers):
            getattr(self, "conv{}".format(i+1)).to_device(device)
        self.to(device)
        return self


    def forward(self, data, record_Z=False, isplot=False):
        """Main forward function.
        
        Args:
            data: the pytorch-geometric data class.
            record_Z: whether to record the standard deviation for the representation Z.
            isplot:   whether to plot.
        
        Returns:
            x: output
            reg_info: other information or metrics.
        """
        reg_info = {}
        if self.model_type == "GCN":
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            for i in range(self.num_layers - 1):
                layer = getattr(self, "conv{}".format(i + 1))
                x, ixz, structure_kl_loss = layer(x, edge_index, edge_weight)
                # Record:
                record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
                if layer.reparam_mode is not None:
                    record_data(reg_info, [layer.Z_std], ["Z_std"])
                if record_Z:
                    record_data(reg_info, [to_np_array(x)], ["Z_{}".format(i)], nolist=True)
                if self.with_relu:
                    x = F.relu(x)
                    if self.dropout is True:
                        x = F.dropout(x, training=self.training)
            layer = getattr(self, "conv{}".format(self.num_layers))
            x, ixz, structure_kl_loss = layer(x, edge_index, edge_weight)
            # Record:
            record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
            if layer.reparam_mode is not None:
                record_data(reg_info, [layer.Z_std], ["Z_std"])
            if record_Z:
                record_data(reg_info, [to_np_array(x)], ["Z_{}".format(self.num_layers - 1)], nolist=True)

        elif self.model_type == "GAT":
            x = F.dropout(data.x, p=0.6, training=self.training)

            for i in range(self.num_layers - 1):
                if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                    x_1, ixz_1, structure_kl_loss_1, _ = getattr(self, "conv{}_1".format(i + 1))(x, data.multi_edge_index)
                layer = getattr(self, "conv{}".format(i + 1))
                x, ixz, structure_kl_loss, _ = layer(x, data.edge_index)
                # Record:
                record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
                if layer.reparam_mode is not None:
                    record_data(reg_info, [layer.Z_std], ["Z_std"])
                if record_Z:
                    record_data(reg_info, [to_np_array(x)], ["Z_{}".format(i)], nolist=True)
                # Multi-hop:
                if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                    x = torch.cat([x, x_1], dim=-1)
                    record_data(reg_info, [ixz_1, structure_kl_loss_1], ["ixz_DN_list", "structure_kl_DN_list"])
                x = F.elu(x)
                x = F.dropout(x, p=0.6, training=self.training)

            if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                x_1, ixz_1, structure_kl_loss_1, (mu1, std1) = getattr(self, "conv{}_1".format(self.num_layers))(x, data.multi_edge_index)
            layer = getattr(self, "conv{}".format(self.num_layers))
            x, ixz, structure_kl_loss, (mu, std) = layer(x, data.edge_index)
            # Record:
            record_data(reg_info, [ixz, structure_kl_loss], ["ixz_list", "structure_kl_list"])
            if layer.reparam_mode is not None:
                record_data(reg_info, [layer.Z_std], ["Z_std"])
            if record_Z:
                record_data(reg_info, [to_np_array(x)], ["Z_{}".format(self.num_layers - 1)], nolist=True)
            # Multi-hop:
            if self.struct_dropout_mode[0] == 'DNsampling' or (self.struct_dropout_mode[0] == 'standard' and len(self.struct_dropout_mode) == 3):
                x = x + x_1
                record_data(reg_info, [ixz_1, structure_kl_loss_1], ["ixz_DN_list", "structure_kl_DN_list"])

        return x, reg_info, ((mu+mu1)/2, (std+std1)/2)


    def compute_metrics_fun(self, data, metrics, mask=None, mask_id=None):
        """Compute metrics for measuring clustering performance.
        Choices: "Silu", "CH", "DB".
        """
        _, info_dict = self(data, record_Z=True)
        y = to_np_array(data.y)
        info_metrics = {}
        if mask is not None:
            mask = to_np_array(mask)
            mask_id += "_"
        else:
            mask_id = ""
        for k in range(self.num_layers):
            if mask is not None:
                Z_i = info_dict["Z_{}".format(k)][mask]
                y_i = y[mask]
            else:
                Z_i = info_dict["Z_{}".format(k)]
                y_i = y
            for metric in metrics:
                if metric == "Silu":
                    score = sklearn.metrics.silhouette_score(Z_i, y_i, metric='euclidean')
                elif metric == "DB":
                    score = sklearn.metrics.davies_bouldin_score(Z_i, y_i)
                elif metric == "CH":
                    score = sklearn.metrics.calinski_harabasz_score(Z_i, y_i)
                info_metrics["{}{}_{}".format(mask_id, metric, k)] = score
        return info_metrics

class robust_GIB(torch.nn.Module):
    def __init__(
        self,
        model_type,
        num_features,
        num_classes,
        reparam_mode,
        prior_mode,
        latent_size,
        sample_size=1,
        num_layers=2,
        struct_dropout_mode=("standard", 0.6),
        dropout=True,
        with_relu=True,
        val_use_mean=True,
        reparam_all_layers=True,
        normalize=True,
        is_cuda=False,
    ):
        super(robust_GIB, self).__init__()
        self.struct_dropout_mode = struct_dropout_mode
        self.encoder_S = GNN(
            model_type=model_type,
            num_features=num_features,
            num_classes=latent_size,
            reparam_mode=reparam_mode,
            prior_mode=prior_mode,
            latent_size=latent_size,
            sample_size=sample_size,
            num_layers=num_layers,
            struct_dropout_mode=struct_dropout_mode,
            dropout=dropout,
            with_relu=with_relu,
            val_use_mean=val_use_mean,
            reparam_all_layers=reparam_all_layers,
            normalize=normalize,
            is_cuda=is_cuda
        )
        self.encoder_T = GNN(
            model_type=model_type,
            num_features=num_features,
            num_classes=latent_size,
            reparam_mode=reparam_mode,
            prior_mode=prior_mode,
            latent_size=latent_size,
            sample_size=sample_size,
            num_layers=num_layers,
            struct_dropout_mode=struct_dropout_mode,
            dropout=dropout,
            with_relu=with_relu,
            val_use_mean=val_use_mean,
            reparam_all_layers=reparam_all_layers,
            normalize=normalize,
            is_cuda=is_cuda
        )
        self.decoder = nn.Linear(latent_size, num_classes)

    def forward(self, data):
        x_S, reg_info_S, (mu_S, std_S) = self.encoder_S(data)
        x_T, reg_info_T, (mu_T, std_T) = self.encoder_T(data)
        x_S = self.decoder(x_S)
        x_T = self.decoder(x_T)
        
        return reg_info_S, (mu_S, std_S), x_S, reg_info_T, (mu_T, std_T), x_T


class Discriminator(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=2, num_classes=10):
        super(Discriminator, self).__init__()
        
        self.linear_x = nn.Linear(input_dim, hidden_dim)
        self.linear_y = nn.Linear(num_classes, hidden_dim)
        self.linear_sum = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, y, xx, yy):
        h_x = F.relu(self.linear_x(x))
        h_y = F.relu(self.linear_y(y))
        h1 = self.linear_sum(torch.cat([h_x, h_y], dim=1))

        h_xx = F.relu(self.linear_x(xx))
        h_yy = F.relu(self.linear_y(yy))
        h2 = self.linear_sum(torch.cat([h_xx, h_yy], dim=1))

        h = torch.cat([h1, h2], dim=1)
        out = self.classifier(h)

        return out