
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_scale_lengths(seq_length, num_scales):
    return [max(1, seq_length // (2 ** k)) for k in range(num_scales)]


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class PyramidLayer(nn.Module):

    def __init__(self, c_in, c_out, k_size, downsample=True):
        super(PyramidLayer, self).__init__()
        self.norm_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.rec_conv = nn.Conv2d(
            c_in, c_out, kernel_size=(1, k_size),
            padding=(0, (k_size - 1) // 2)
        )
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.relu = nn.ReLU()

    def forward(self, x, target_len):
        x_norm = self.norm_conv(x)
        x_rec = self.rec_conv(x)
        if self.downsample:
            x_rec = self.pool(x_rec)
        t = x_rec.size(3)
        x_norm = x_norm[..., -t:]
        out = self.relu(x_rec + x_norm)
        if out.size(3) > target_len:
            out = out[..., -target_len:]
        elif out.size(3) < target_len:
            out = F.pad(out, (target_len - out.size(3), 0))
        return out


class multi_scale_block(nn.Module):

    def __init__(self, c_in, c_out, seq_length, layer_num, kernel_set):
        super(multi_scale_block, self).__init__()
        self.layer_num = layer_num
        self.target_lengths = compute_scale_lengths(seq_length, layer_num)
        self.start_conv = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(
                PyramidLayer(c_out, c_out, kernel_set[i], downsample=(i > 0))
            )

    def forward(self, input, idx=None):
        x = self.start_conv(input)
        scale_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, self.target_lengths[i])
            scale_outputs.append(x)
        return scale_outputs


class adaptive_graph_learning(nn.Module):

    def __init__(self, nnodes, k, dim, num_scales, device, beta=3):
        super(adaptive_graph_learning, self).__init__()
        self.nnodes = nnodes
        self.k = k
        self.dim = dim
        self.num_scales = num_scales
        self.beta = beta
        self.device = device

        self.node_emb = nn.Embedding(nnodes, dim)
        self.scale_emb = nn.Embedding(num_scales, dim)
        self.delta = nn.Parameter(torch.ones(num_scales))
        self.omega = nn.Parameter(torch.ones(num_scales))

    def forward(self, idx):
        node_e = self.node_emb(idx)
        adj_set = []
        for k in range(self.num_scales):
            scale_e = self.scale_emb(
                torch.tensor(k, device=idx.device, dtype=torch.long)
            )
            e_spec = node_e * scale_e.unsqueeze(0)
            m1 = torch.tanh(e_spec * self.delta[k])
            m2 = torch.tanh(e_spec * self.omega[k])
            a_raw = torch.mm(m1, m2.t()) - torch.mm(m2, m1.t())
            a_full = F.relu(torch.tanh(self.beta * a_raw))

            mask = torch.zeros(idx.size(0), idx.size(0), device=idx.device)
            _, top_idx = a_full.topk(min(self.k, idx.size(0)), dim=1)
            mask.scatter_(1, top_idx, 1.0)
            adj_set.append(a_full * mask)
        return adj_set


graph_constructor = adaptive_graph_learning


class diffusion_gnn(nn.Module):

    def __init__(self, c_in, c_out, gdep, alpha):
        super(diffusion_gnn, self).__init__()
        self.nconv = nconv()
        self.w_in = linear(c_in, c_in)
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.alpha = alpha

    @staticmethod
    def _row_normalize(adj):
        d = adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return adj / d

    def forward(self, x, adj):
        x_in = self.w_in(x)
        a_hat = self._row_normalize(adj)
        a_hat_t = self._row_normalize(adj.t())
        h = x
        out = [h]
        for _ in range(self.gdep):
            prop = self.nconv(h, a_hat) + self.nconv(h, a_hat_t)
            h = self.alpha * x_in + (1.0 - self.alpha) * prop
            out.append(h)
        return self.mlp(torch.cat(out, dim=1))


class gated_tcn(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super(gated_tcn, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv_tanh = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size), padding=(0, pad)
        )
        self.conv_sig = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size), padding=(0, pad)
        )

    def forward(self, x):
        return torch.tanh(self.conv_tanh(x)) * torch.sigmoid(self.conv_sig(x))


class scale_fusion_network(nn.Module):

    def __init__(self, channels, num_scales, ratio=1):
        super(scale_fusion_network, self).__init__()
        self.num_scales = num_scales
        self.fc1 = nn.Linear(channels, channels * ratio)
        self.fc2 = nn.Linear(channels * ratio, num_scales)

    def forward(self, h_list):
        h_stack = torch.stack(h_list, dim=1)
        h_pool = h_stack.mean(dim=3).mean(dim=1).squeeze(-1)
        alpha1 = F.relu(self.fc1(h_pool))
        weights = torch.sigmoid(self.fc2(alpha1))
        weights = weights.view(weights.size(0), self.num_scales, 1, 1, 1)
        h_m = (h_stack * weights).sum(dim=1)
        return F.relu(h_m)


gated_fusion = scale_fusion_network


class mixprop(nn.Module):

    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.gnn = diffusion_gnn(c_in, c_out, gdep, alpha)
        self.dropout = dropout

    def forward(self, x, adj):
        if self.training and self.dropout > 0:
            x = F.dropout(x, self.dropout, training=True)
        return self.gnn(x, adj)
