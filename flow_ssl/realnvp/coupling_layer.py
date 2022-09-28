import torch
import torch.nn as nn

from enum import IntEnum
from flow_ssl.resnet_realnvp import ResNet
from flow_ssl.realnvp.utils import checkerboard_mask
import torch.nn.functional as F
from flow_ssl.gcn.gcnmodels import *
import numpy as np

class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1
    TABULAR = 2


class MaskChannelwise:
    def __init__(self, reverse_mask):
        self.type = MaskType.CHANNEL_WISE
        self.reverse_mask = reverse_mask

    def mask(self, x):
        if self.reverse_mask:
            x_id, x_change = x.chunk(2, dim=1)
        else:
            x_change, x_id = x.chunk(2, dim=1)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        if self.reverse_mask:
            return torch.cat((x_id, x_change), dim=1)
        else:
            return torch.cat((x_change, x_id), dim=1)
        
    def mask_st_output(self, s, t):
        return s, t


class MaskCheckerboard:
    def __init__(self, reverse_mask):
        self.type = MaskType.CHECKERBOARD
        self.reverse_mask = reverse_mask

    def mask(self, x):
        self.b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)


class MaskTabular:
    def __init__(self, reverse_mask):
        self.type = MaskType.TABULAR
        self.reverse_mask = reverse_mask


    def mask(self, x):
        #PAVEL: should we pass x sizes to __init__ and only create mask once?
        dim = x.size(1)
        split = dim // 2
        self.b = torch.zeros((1, dim), dtype=torch.float).to(x.device)
        #selected_dim = np.random.choice(range(dim), split, replace=False)
        #selected_dim = np.sort(selected_dim)

        #all_dim = np.array([i for i in range(0, dim)])
        #selected_dim = np.array([i for i in range(0, dim, 2)])
        #remain_dim = np.array(list(set(all_dim).difference(set(selected_dim))))
        if self.reverse_mask:
            self.b[:, split:] = 1.
            #self.b[:,remain_dim]=1.

        else:
            self.b[:, :split] = 1.
            #self.b[:,selected_dim]=1.
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)


class CouplingLayerBase(nn.Module):
    """Coupling layer base class in RealNVP.
    
    must define self.mask, self.st_net, self.rescale
    """
    def _get_st(self, x):
        x_id, x_change = self.mask.mask(x)
        x_id_adj = torch.eye(x_id.shape[0])
        x_x_adj_id=(x_id, x_id_adj)
        st = self.st_net(x_x_adj_id)
        s, t = st.chunk(2, dim=1)
        s = self.rescale(torch.tanh(s))
        return s, t, x_id, x_change

    # def forward(self, x, sldj=None, reverse=True):
    #     s, t, x_id, x_change = self._get_st(x)
    #     s, t = self.mask.mask_st_output(s, t)
    #     exp_s = s.exp()
    #     if torch.isnan(exp_s).any():
    #         raise RuntimeError('Scale factor has NaN entries')
    #     x_change = (x_change + t) * exp_s
    #     self._logdet = s.view(s.size(0), -1).sum(-1)
    #     x = self.mask.unmask(x_id, x_change)
    #     return x

    def forward(self, x_adj_mat, sldj=None, reverse=True):
        if isinstance(x_adj_mat,tuple):
            x=x_adj_mat[0]
            adj_mat = x_adj_mat[1]
        x=torch.spmm(adj_mat, x)
        s, t, x_id, x_change = self._get_st(x)
        s, t = self.mask.mask_st_output(s, t)
        exp_s = s.exp()
        if torch.isnan(exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = (x_change + t) * exp_s
        #sample_num=s.view(s.size(0), -1).shape[0]
        self._logdet = s.view(s.size(0), -1).sum()
        self._matdet = s.view(s.size(0), -1)
        #self._logdet = s.view(s.size(0), -1).sum(-1)
        x = self.mask.unmask(x_id, x_change)
        return (x,adj_mat)

    def inverse(self, y):
        s, t, x_id, x_change = self._get_st(y)
        s, t = self.mask.mask_st_output(s, t)
        exp_s = s.exp()
        inv_exp_s = s.mul(-1).exp()
        if torch.isnan(inv_exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = x_change * inv_exp_s - t
#         self._logdet = -s.view(s.size(0), -1).sum(-1)
        x = self.mask.unmask(x_id, x_change)
        return x

    def logdet(self):
        return self._logdet

    def matdet(self):
        return self._matdet


class CouplingLayer(CouplingLayerBase):
    """Coupling layer in RealNVP for image data.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask (MaskChannelWise or MaskChannelWise): mask.
    """

    def __init__(self, in_channels, mid_channels, num_blocks, mask, init_zeros=False):
        super(CouplingLayer, self).__init__()

        self.mask = mask

        # Build scale and translate network
        if self.mask.type == MaskType.CHANNEL_WISE:
            in_channels //= 2

        # Pavel: reuse Marc's ResNet block?
        self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
                             num_blocks=num_blocks, kernel_size=3, padding=1,
                             double_after_norm=(self.mask.type == MaskType.CHECKERBOARD),
                             init_zeros=init_zeros)

        # Learnable scale for s
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x


class CouplingLayerTabular(CouplingLayerBase):

    def __init__(self, in_dim, mid_dim, num_layers, mask, init_zeros=False,dropout=False):

        super(CouplingLayerTabular, self).__init__()
        self.mask = mask
        self.layers = [
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(.5) if dropout else nn.Sequential(),
            *self._inner_seq(num_layers, mid_dim),
        ]
        last_layer = nn.Linear(mid_dim, in_dim*2)
        if init_zeros:
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)
        self.layers.append(last_layer)

        self.st_net = nn.Sequential(*self.layers)
        self.rescale = nn.utils.weight_norm(RescaleTabular(in_dim))
       
    @staticmethod
    def _inner_seq(num_layers, mid_dim):
        res = []
        for _ in range(num_layers):
            res.append(nn.Linear(mid_dim, mid_dim))
            res.append(nn.ReLU())
        return res


class RescaleTabular(nn.Module):
    def __init__(self, D):
        super(RescaleTabular, self).__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        x = self.weight * x
        return x

class CouplingLayerGraph(CouplingLayerBase):
    def __init__(self, in_dim, mid_dim, mask,
                 nhiddenlayer=0,
                 dropout_ratio=0.5,
                 baseblocktype="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaseblocklayer=0,
                 aggrmethod="add",
                 withbn=False,
                 withloop=False,
                 init_zeros=False):
        super(CouplingLayerGraph, self).__init__()
        self.mask = mask
        self.st_net = GCNModel(nfeat=in_dim,
                               nhid=mid_dim,
                               nclass=in_dim*2,
                               nhidlayer=nhiddenlayer,
                               dropout_ratio=dropout_ratio,
                               baseblock=baseblocktype,
                               inputlayer=inputlayer,
                               outputlayer=outputlayer,
                               nbaselayer=nbaseblocklayer,
                               activation=F.relu,
                               withbn=withbn,
                               withloop=withloop,
                               aggrmethod=aggrmethod)
        self.rescale = nn.utils.weight_norm(RescaleGraph(in_dim))

class RescaleGraph(nn.Module):
    def __init__(self, D):
        super(RescaleGraph, self).__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        x = self.weight * x
        return x