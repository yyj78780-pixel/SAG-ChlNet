
from layer import *
import torch
import torch.nn.functional as F
from torch import nn


class magnn(nn.Module):

    def __init__(self,
                 gcn_depth,
                 num_nodes,
                 device,
                 dropout=0.1,
                 subgraph_size=20,
                 node_dim=40,
                 conv_channels=64,
                 gnn_channels=32,
                 scale_channels=16,
                 end_channels=128,
                 seq_length=30,
                 in_dim=1,
                 out_dim=12,
                 num_scales=4,
                 propalpha=0.05,
                 tanhalpha=3,
                 single_step=False,
                 use_scale0=True):
        super(magnn, self).__init__()

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.device = device
        self.seq_length = seq_length
        self.use_scale0 = use_scale0

        if single_step:
            base_kernel_set = [15, 10, 7, 5, 3, 2]
        else:
            base_kernel_set = [10, 7, 5, 3, 2, 2]

        self.num_scales = min(num_scales, len(base_kernel_set))
        self.kernel_set = base_kernel_set[:self.num_scales]
        self.scale_lengths = compute_scale_lengths(seq_length, self.num_scales)

        print(f"[MAGNN] K={self.num_scales}, kernels={self.kernel_set}, "
              f"T_k={self.scale_lengths}, scale0={self.use_scale0}")

        self.idx = torch.arange(self.num_nodes, device=device)

        self.mspn = multi_scale_block(
            in_dim, conv_channels, seq_length, self.num_scales, self.kernel_set
        )
        if self.use_scale0:
            self.scale0 = nn.Conv2d(
                in_dim, scale_channels, kernel_size=(1, seq_length), bias=True
            )
        self.agl = adaptive_graph_learning(
            num_nodes, subgraph_size, node_dim, self.num_scales, device, beta=tanhalpha
        )

        self.diffusion_gnns = nn.ModuleList()
        self.gated_tcns = nn.ModuleList()
        self.temporal_compress = nn.ModuleList()

        for i in range(self.num_scales):
            self.diffusion_gnns.append(
                diffusion_gnn(conv_channels, gnn_channels, gcn_depth, propalpha)
            )
            self.gated_tcns.append(gated_tcn(gnn_channels))
            t_k = self.scale_lengths[i]
            self.temporal_compress.append(
                nn.Conv2d(gnn_channels, scale_channels, kernel_size=(1, t_k))
            )

        fusion_scales = self.num_scales + (1 if self.use_scale0 else 0)
        self.sfn = scale_fusion_network(scale_channels, fusion_scales)
        self.end_conv_1 = nn.Conv2d(scale_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, (
            f'Input sequence length {seq_len} != preset {self.seq_length}'
        )

        if idx is None:
            idx = self.idx

        x = F.dropout(input, self.dropout, training=self.training)

        scale_features = self.mspn(x, idx)
        adj_matrix = self.agl(self.idx)
        adj_matrix_sub = [adj[idx][:, idx] for adj in adj_matrix]

        h_list = []
        if self.use_scale0:
            h_list.append(self.scale0(x))
        for i in range(self.num_scales):
            h = self.diffusion_gnns[i](scale_features[i], adj_matrix_sub[i])
            h = self.gated_tcns[i](h)
            t_len = h.size(3)
            if t_len != self.scale_lengths[i]:
                h = h[..., -self.scale_lengths[i]:]
            h = self.temporal_compress[i](h)
            h_list.append(h)

        h_m = self.sfn(h_list)
        out = self.end_conv_2(F.relu(self.end_conv_1(h_m)))
        return out, adj_matrix_sub

