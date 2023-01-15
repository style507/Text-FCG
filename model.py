import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import nn as gnn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from typing import Optional
from torch_geometric.typing import OptTensor
from torch_geometric.nn import GATConv

from optional_gat_conv import OpGATConv
from film_conv import FiLMConv
# from gat_conv import GATConv

import math

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class GRUUint(nn.Module):

    def __init__(self, hid_dim, act):
        super(GRUUint, self).__init__()
        self.act = act
        self.lin_z0 = nn.Linear(hid_dim, hid_dim)
        self.lin_z1 = nn.Linear(hid_dim, hid_dim)
        self.lin_r0 = nn.Linear(hid_dim, hid_dim)
        self.lin_r1 = nn.Linear(hid_dim, hid_dim)
        self.lin_h0 = nn.Linear(hid_dim, hid_dim)
        self.lin_h1 = nn.Linear(hid_dim, hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, a):
        z = (self.lin_z0(a) + self.lin_z1(x)).sigmoid()
        r = (self.lin_r0(a) + self.lin_r1(x)).sigmoid()
        h = self.act((self.lin_h0(a) + self.lin_h1(x * r)))
        return h * z + x * (1 - z)


class GRUUint_s(nn.Module):

    def __init__(self, hid_dim, act):
        super(GRUUint_s, self).__init__()
        self.act = act
        self.lin_z0 = nn.Linear(hid_dim, hid_dim)
        self.lin_z1 = nn.Linear(hid_dim, hid_dim)
        self.lin_z2 = nn.Linear(hid_dim, hid_dim)

        self.lin_r0 = nn.Linear(hid_dim, hid_dim)
        self.lin_r1 = nn.Linear(hid_dim, hid_dim)
        self.lin_r2 = nn.Linear(hid_dim, hid_dim)

        self.lin_h0 = nn.Linear(hid_dim, hid_dim)
        self.lin_h1 = nn.Linear(hid_dim, hid_dim)
        self.lin_h2 = nn.Linear(hid_dim, hid_dim)

        self.W0 = nn.Linear(hid_dim, hid_dim)
        self.w0 = nn.Linear(hid_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, a, s):
        z = (self.lin_z0(a) + self.lin_z1(x) + self.lin_z2(s)).sigmoid()
        r = (self.lin_r0(a) + self.lin_r1(x) + self.lin_r2(s)).sigmoid()
        h = self.act((self.lin_h0(a) + self.lin_h1(x * r) + self.lin_h2(s)))
        return h * z + x * (1 - z)

    # def forward(self, x, a, s):
    #     gates = torch.stack((a, s), dim=1)
    #     new_a = self.attention(gates)
    #     z = (self.lin_z0(new_a) + self.lin_z1(x)).sigmoid()
    #     r = (self.lin_r0(new_a) + self.lin_r1(x)).sigmoid()
    #     h = self.act((self.lin_h0(new_a) + self.lin_h1(x * r)))
    #     return h * z + x * (1 - z)

    def attention(self, H):
        # c: batch_size, seq_len, embedding_dim
        c = self.w0(torch.tanh(self.W0(H)))  # batch_size, seq_len, 1
        atten = F.softmax(c.squeeze(-1), -1)  # batch_size, seq_len
        A = torch.bmm(atten.unsqueeze(1), H).squeeze(1)
        return A


class GraphAT(torch.nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.5, act=torch.relu, bias=True, step=2):
        super(GraphAT, self).__init__()

        self.step = step
        self.act = act
        self.dropout = dropout

        self.encode = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True),
        )
        # self.conv = OpGATConv(out_dim, out_dim, heads=3, dropout=0.5, negative_slope=0.2,
        #                       add_self_loops=True, concat=False, bias=bias, attention_type='MX',
        #                       root_weight=False)
        # self.conv = GATConv(out_dim, out_dim, heads=3, dropout=0.5, negative_slope=0.2,
        #                     add_self_loops=True, concat=False, bias=bias)

        # self.conv = FiLMConv(out_dim, out_dim, num_relations=4)
        self.conv = nn.ModuleList()
        # self.conv.append(FiLMConv(in_dim, out_dim, num_relations=4, act=None))
        for _ in range(self.step):
            self.conv.append(FiLMConv(out_dim, out_dim, num_relations=4))   # 消融

        self.gru = GRUUint(out_dim, act=act)
        self.gru_s = GRUUint_s(out_dim, act=act)
        self.W2 = nn.Linear(out_dim, out_dim)
        self.w2 = nn.Linear(out_dim, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, g, mask):
        x = self.encode(x)
        x = self.act(x)

        for i in range(self.step):
            sen = self.graph2batch(x, g.length)
            senten = self.self_attention(sen, mask)
            sn = self.sentnece2node(senten, g.length)

            a = self.conv[i](x, g.edge_index, g.edge_type)
            # a = self.conv(x, g.edge_index)
            x = self.gru_s(x, a, sn)
            # x = self.gru(x, a)

        x = self.graph2batch(x, g.length)
        return x

    def graph2batch(self, x, length):
        x_list = []
        for l in length:
            x_list.append(x[:l])
            x = x[l:]
        x = pad_sequence(x_list, batch_first=True)
        return x

    def sentnece2node(self, s, length):
        node_list = []
        for l in range(0, len(length)):
            temp = s[l].repeat(length[l], 1)
            node_list.append(temp)
        node_list = torch.cat(node_list, 0)
        # if node_list.is_cuda:
        #     node_list = node_list.to(device)
        return node_list

    def attention(self, H, mask):
        # c: batch_size, seq_len, embedding_dim
        H = H * mask
        c = self.w2(torch.tanh(self.W2(H)))  # batch_size, seq_len, 1
        atten = F.softmax(c.squeeze(-1), -1)  # batch_size, seq_len
        A = torch.bmm(atten.unsqueeze(1), H).squeeze(1)
        return A

    def self_attention(self, H, mask):
        # H: batch_size, seq_len, 2*hidden_size
        H = H * mask
        hidden_size = H.size()[-1]
        Q = H
        K = H
        V = H
        # batch_size, seq_len, seq_len
        atten_weight = F.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(hidden_size), -1)
        A = torch.bmm(atten_weight, V)  # batch_size, seq_len, 2*hidden_size
        A = A.permute(0, 2, 1)
        # q: text representation
        # -> avg+add max+add max+mean avg+mean
        q = F.avg_pool1d(A, A.size()[2]).squeeze(-1)  # batch_size, 2*hidden_size
        # q = torch.mean(A, dim=-1)
        return q


class GraphAttentionLayer(gnn.MessagePassing):

    def __init__(self, in_dim, out_dim, dropout=0.5, act=torch.relu, bias=False, step=2,
                 requires_grad=True, add_self_loops=False):
        super(GraphAttentionLayer, self).__init__(aggr='add')
        self.step = step
        self.act = act

        self.requires_grad = requires_grad
        self.add_self_loops = add_self_loops

        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.encode = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.gru = GRUUint(out_dim, act=act)
        # self.gru = GRUUint_s(out_dim, act=act)
        self.dropout = nn.Dropout(dropout)

        self.W2 = nn.Linear(in_dim, in_dim)
        self.w2 = nn.Linear(in_dim, 1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, g):
        x = self.encode(x)
        x = self.act(x)

        edge_index = g.edge_index
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))

        x_norm = F.normalize(x, p=2., dim=-1)
        for _ in range(self.step):
            # x_norm = F.normalize(x, p=2., dim=-1)
            a = self.propagate(edge_index=edge_index, x=x, x_norm=x_norm, size=None)

            # a2batch = self.graph2batch(a, g.length)
            # s = self.attention(a2batch)
            # sn = self.sentnece2node(s, g.length)
            # x = self.gru(x, a, sn)

            x = self.gru(x, a)
        x = self.graph2batch(x, g.length)

        return x

    def message(self, x_j: Tensor, x_norm_i: Tensor, x_norm_j: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = self.beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha.view(-1, 1)

    def update(self, inputs):
        return inputs

    def graph2batch(self, x, length):
        x_list = []
        for l in length:
            x_list.append(x[:l])
            x = x[l:]
        x = pad_sequence(x_list, batch_first=True)
        return x

    def sentnece2node(self, s, length):
        node_list = []
        for l in range(0, len(length)):
            temp = s[l].repeat(length[l], 1)
            node_list.append(temp)
        node_list = torch.cat(node_list, 0)

        if node_list.is_cuda:
            node_list = node_list.to(device)
        return node_list

    def self_attention(self, H):
        # H: batch_size, seq_len, 2*hidden_size
        hidden_size = H.size()[-1]
        Q = H
        K = H
        V = H
        # batch_size, seq_len, seq_len
        atten_weight = F.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(hidden_size), -1)
        A = torch.bmm(atten_weight, V)  # batch_size, seq_len, 2*hidden_size
        A = A.permute(0, 2, 1)
        # q: text representation
        q = F.max_pool1d(A, A.size()[2]).squeeze(-1)  # batch_size, 2*hidden_size
        return q

    def attention(self, H):
        # c: batch_size, seq_len, embedding_dim
        c = self.w2(torch.tanh(self.W2(H)))  # batch_size, seq_len, 1
        atten = F.softmax(c.squeeze(-1), -1)  # batch_size, seq_len
        A = torch.bmm(atten.unsqueeze(1), H).squeeze(1)
        return A


class BiRNN(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.5, bias=False, bidirectional=True):
        super(BiRNN, self).__init__()

        self.bias = bias
        self.hid_dim = hid_dim

        self.encode = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=True)
        )
        self.rnn = nn.LSTM(in_dim, hid_dim, batch_first=True, bidirectional=bidirectional)

        self.dropout = dropout
        self.reset_parameters()

        self.W1 = nn.Linear(hid_dim, hid_dim)
        self.w1 = nn.Linear(hid_dim, 1, bias=True)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        out, (final_hidden, _) = self.rnn(x)

        # hidden = final_hidden.view(-1, self.hid_dim * 2, 1)
        # attn_weights = torch.bmm(out, hidden).squeeze(2)
        # atten = F.softmax(attn_weights, 1)
        # out = torch.bmm(out.transpose(1, 2), atten.unsqueeze(2)).squeeze(2)

        # out = F.dropout(out, self.dropout, training=self.training)
        # x = self.encode(x)

        # 将双向lstm的输出拆分为前向输出和后向输出
        # (forward_out, backward_out) = torch.chunk(out, 2, -1)
        # out = torch.cat((forward_out, x, backward_out), 2)  # 100*3

        # attention
        # gates = torch.stack((forward_out, x, backward_out), dim=1)
        # sen = gates.shape[2]
        # inputs = gates.permute(2, 0, 1, 3)
        # outputs = []
        # for i in range(sen):
        #     t_x = inputs[i, :, :, :]
        #
        #     # output = self.self_attention(t_x)
        #     # output = torch.sum(output, dim=1)
        #
        #     output = torch.tanh(self.W1(t_x)) * self.w1(t_x).sigmoid()
        #     output = torch.reshape(output, [output.shape[0], output.shape[1] * output.shape[-1]])
        #
        #     outputs.append(output)
        # out = torch.stack(outputs, axis=1)

        return out

    def self_attention(self, H):
        # H: batch_size, seq_len, 2*hidden_size
        hidden_size = H.size()[-1]
        Q = H
        K = H
        V = H
        # batch_size, seq_len, seq_len
        atten_weight = F.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(hidden_size), -1)
        A = torch.bmm(atten_weight, V)  # batch_size, seq_len, 2*hidden_size
        A = A.permute(0, 2, 1)

        # q: text representation
        # q = F.avg_pool1d(A, A.size()[2]).squeeze(-1)  # batch_size, 2*hidden_size
        # q = torch.mean(A, dim=-1)

        q = torch.reshape(A, [A.shape[0], A.shape[1] * A.shape[-1]])
        return q

    def attention(self, H):
        # c: batch_size, seq_len, embedding_dim
        c = self.w1(torch.tanh(self.W1(H)))  # batch_size, seq_len, 1
        atten = F.softmax(c.squeeze(-1), -1)  # batch_size, seq_len
        A = torch.bmm(atten.unsqueeze(1), H).squeeze(1)
        return A


class ReadoutLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.5,
                 act=torch.relu, bias=False):
        super(ReadoutLayer, self).__init__()
        self.act = act
        self.bias = bias
        self.att = nn.Linear(in_dim, 1, bias=True)
        self.emb = nn.Linear(in_dim, in_dim, bias=True)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, out_dim, bias=True)
        )
        self.w3 = nn.Linear(in_dim, 1, bias=True)
        self.W3 = nn.Linear(in_dim, in_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mask):
        # emb = self.act(self.emb(x))
        # att = self.att(x).sigmoid()
        # x = att * emb
        # x = self.__max(x, mask) + self.__mean(x, mask)

        x = self.attention(x, mask)
        x = self.mlp(x)
        return x

    # def attention(self, H, mask):
    #     # batch_size, seq_len, embedding_dim
    #     emb = torch.tanh(self.W3(H))
    #     # batch_size, seq_len, 1
    #     c = self.w3(emb)
    #     # batch_size, seq_len
    #     atten = F.softmax(c.squeeze(-1), -1)
    #     A = torch.bmm(atten.unsqueeze(1), H * mask).squeeze(1)
    #     B = atten.unsqueeze(-1) * H
    #     return A, B

    def attention(self, H, mask):
        # c: batch_size, seq_len, embedding_dim
        H = H * mask
        c = self.w3(torch.tanh(self.W3(H)))  # batch_size, seq_len, 1
        atten = F.softmax(c.squeeze(-1), -1)  # batch_size, seq_len
        A = torch.bmm(atten.unsqueeze(1), H).squeeze(1)
        return A

    def __max(self, x, mask):
        return (x + (mask - 1) * 1e9).max(1)[0]

    def __mean(self, x, mask):
        return (x * mask).sum(1) / mask.sum(1)


class Model(nn.Module):
    def __init__(self, num_words, num_classes, in_dim=100, hid_dim=100, step=2, word2vec=None, freeze=True):
        super(Model, self).__init__()
        if word2vec is None:
            self.embed = nn.Embedding(num_words + 1, in_dim, num_words)
        else:
            self.embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(word2vec).float(), freeze, num_words)

        # self.gcn = GraphAttentionLayer(hid_dim * 2, hid_dim * 2, act=torch.tanh, step=step)
        # back
        self.gcn = GraphAT(in_dim, hid_dim, act=torch.tanh, step=step)
        # back
        self.bilstm = BiRNN(hid_dim, hid_dim)
        # back
        self.read = ReadoutLayer(hid_dim * 2, num_classes, act=torch.tanh)
        # self.read = ReadoutLayer(hid_dim, num_classes, act=torch.tanh)

        self.encode = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, g):
        mask = self.get_mask(g)

        # x = self.word2batch(g.x, g.length)
        # x = self.embed(x)
        # x = self.bilstm(x)
        # x = self.batch2word(x, g.length)

        x = self.embed(g.x)
        # x = self.gcn(x, g)
        x = self.gcn(x, g, mask)
        x = self.bilstm(x)
        # x = self.encode(x)
        x = self.read(x, mask)
        return x

    def get_mask(self, g):
        mask = pad_sequence([torch.ones(l) for l in g.length], batch_first=True).unsqueeze(-1)
        if g.x.is_cuda:
            mask = mask.to(device)
        return mask

    def word2batch(self, x, length):
        x_list = []
        for l in length:
            x_list.append(x[:l])
            x = x[l:]
        x = pad_sequence(x_list, batch_first=True)
        return x

    def batch2word(self, batch, length):
        x_list = []
        for l in range(0, len(length)):
            temp = batch[l][0:length[l]]
            x_list.append(temp)
        x_tensor = torch.cat(x_list, 0)
        return x_tensor
