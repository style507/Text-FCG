from typing import Optional

import math
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class OpGATConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0, add_self_loops: bool = True,
                 bias: bool = True, attention_type: str = 'MX',  root_weight: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(OpGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.root_weight = root_weight

        assert attention_type in ['MX', 'SD']

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))

        if self.attention_type == 'MX':
            self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
            self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        else:  # self.attention_type == 'SD'
            self.register_parameter('att_l', None)
            self.register_parameter('att_r', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.root_weight:
            self.lin_r = Linear(in_channels, out_channels, bias=False)
        else:
            self.register_parameter('lin_r', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)
        if self.lin_r is not None:
            self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        N, H, C = x.size(0), self.heads, self.out_channels

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        x_r = x[1]
        x = torch.matmul(x, self.weight).view(-1, H, C)

        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x, size=None)

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor, size_i: Optional[int]) -> Tensor:
        alpha = self.get_attention(edge_index_i, x_i, x_j, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def get_attention(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                      num_nodes: Optional[int],
                      return_logits: bool = False) -> Tensor:

        if self.attention_type == 'MX':
            logits = (x_i * x_j).sum(dim=-1)
            if return_logits:
                return logits

            alpha = (x_j * self.att_l).sum(-1) + (x_i * self.att_r).sum(-1)
            alpha = alpha * logits.sigmoid()

        else:  # self.attention_type == 'SD'
            alpha = (x_i * x_j).sum(dim=-1) / math.sqrt(self.out_channels)
            if return_logits:
                return alpha

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        return alpha
