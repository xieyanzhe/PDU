import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d = num_of_d

    def forward(self, Q, K, V, attn_mask, res_att):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k) + res_att
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, V)
        return context, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k, d_v, n_heads, num_of_d):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        residual, indicators = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(indicators, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2,
                                                                                                    3)
        K = self.W_K(input_K).view(indicators, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2,
                                                                                                    3)
        V = self.W_V(input_V).view(indicators, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2,
                                                                                                    3)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)
        context, res_attn = ScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask, res_att)

        context = context.transpose(2, 3).reshape(indicators, self.num_of_d, -1,
                                                  self.n_heads * self.d_v)
        output = self.fc(context)
        return nn.LayerNorm(self.d_model).to(self.DEVICE)(output + residual), res_attn


class Embedding(nn.Module):
    def __init__(self, nb_seq, d_Em, num_of_features, Etype):
        super(Embedding, self).__init__()
        self.nb_seq = nb_seq
        self.Etype = Etype
        self.num_of_features = num_of_features
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.norm = nn.LayerNorm(d_Em)

    def forward(self, x, indicators):
        if self.Etype == 'T':
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(x.device)
            pos = pos.unsqueeze(0).unsqueeze(0).expand(indicators, self.num_of_features,
                                                       self.nb_seq)
            embedding = x.permute(0, 2, 3, 1) + self.pos_embed(pos)
        else:
            pos = torch.arange(self.nb_seq, dtype=torch.long).to(x.device)
            pos = pos.unsqueeze(0).expand(indicators, self.nb_seq)
            embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)
        return Emx


class GTU(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu


class Predictor(torch.nn.Module):
    def __init__(self, in_feature, out_feature, node_num, hidden_feature=32, headers=4, block_num=3):
        super(Predictor, self).__init__()

        self.hidden_feature = hidden_feature
        self.headers = headers
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.node_num = node_num
        self.block_num = block_num
        self.layer1 = GATv2Conv(self.in_feature, self.hidden_feature, heads=self.headers)
        self.layer2 = GATv2Conv(self.hidden_feature * self.headers, self.hidden_feature, heads=headers)
        self.dense = nn.Linear(self.hidden_feature * self.headers, self.out_feature)
        self.activation = nn.ReLU()

    def forward(self, data, edges):
        features = data
        edges = edges.cuda()
        features = self.layer1(features, edges)
        features = self.activation(features)
        for _ in range(self.block_num):
            features_2 = self.layer2(features, edges)
            features_2 = self.activation(features_2)
            features = features + 0.5 * features_2
        features = self.dense(features)
        return features


class PDU_block(nn.Module):

    def __init__(self, DEVICE, num_of_d, in_channels, nb_time_filter, time_strides,
                 num_of_vertices, num_of_timesteps, d_model, d_k, d_v, n_heads):
        super(PDU_block, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.pre_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, num_of_d))
        self.EmbedT = Embedding(num_of_timesteps, num_of_vertices, num_of_d, 'T')
        self.EmbedS = Embedding(num_of_vertices, d_model, num_of_d, 'S')
        self.TAt = MultiHeadAttention(DEVICE, num_of_vertices, d_k, d_v, n_heads, num_of_d)
        self.gat = GATv2Conv(d_model, 128, heads=1)
        self.gtu3 = GTU(nb_time_filter, time_strides, 2)
        self.gtu5 = GTU(nb_time_filter, time_strides, 3)
        self.gtu7 = GTU(nb_time_filter, time_strides, 4)
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0,
                                          return_indices=False, ceil_mode=False)
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.dropout = nn.Dropout(p=0.02)
        self.fc = nn.Sequential(
            nn.Linear(3 * num_of_timesteps - 4, num_of_timesteps),
            nn.Dropout(0.02),
        )
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, res_att, edges):
        indicators, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        if num_of_features == 1:
            TEmx = self.EmbedT(x, indicators)
        else:
            TEmx = x.permute(0, 2, 3, 1)
        TATout, re_At = self.TAt(TEmx, TEmx, TEmx, None, res_att)
        x_TAt = self.pre_conv(TATout.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)
        SEmx_TAt = self.EmbedS(x_TAt, indicators)
        SEmx_TAt = self.dropout(SEmx_TAt)
        SEmx_TAt = SEmx_TAt.view(num_of_vertices * indicators, -1)
        spatial_gcn = self.gat(SEmx_TAt, edges)
        spatial_gcn1 = spatial_gcn.view(indicators, num_of_vertices, 32, num_of_timesteps)
        X = spatial_gcn1.permute(0, 2, 1, 3)
        x_gtu = []
        x_gtu.append(self.gtu3(X))
        x_gtu.append(self.gtu5(X))
        x_gtu.append(self.gtu7(X))
        time_conv = torch.cat(x_gtu, dim=-1)
        time_conv = self.fc(time_conv)
        if num_of_features == 1:
            time_conv_output = self.relu(time_conv)
        else:
            time_conv_output = self.relu(X + time_conv)
        if num_of_features == 1:
            x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        else:
            x_residual = x.permute(0, 2, 1, 3)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        return x_residual, re_At, SEmx_TAt


class PDU(nn.Module):

    def __init__(self, DEVICE, num_of_d, nb_block, in_channels, nb_chev_filter, nb_time_filter, time_strides,
                 num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v,
                 n_heads):
        super(PDU, self).__init__()
        self.BlockList = nn.ModuleList([PDU_block(DEVICE, num_of_d, in_channels, nb_time_filter,
                                                  time_strides, num_of_vertices, len_input, d_model, d_k, d_v,
                                                  n_heads)])
        self.BlockList.extend([PDU_block(DEVICE, num_of_d * nb_time_filter, nb_chev_filter, nb_time_filter,
                                         1, num_of_vertices, len_input // time_strides, d_model, d_k,
                                         d_v, n_heads) for _ in range(nb_block - 1)])
        self.final_conv = nn.Conv2d(int((len_input / time_strides) * nb_block), num_for_predict,
                                    kernel_size=(1, nb_time_filter))
        self.DEVICE = DEVICE
        self.len_input = len_input
        self.num_for_predict = num_for_predict
        self.norm = nn.BatchNorm1d(num_for_predict)
        self.final_dense = nn.Linear(num_for_predict, num_for_predict)
        self.to(DEVICE)

    def forward(self, data, edge_index):
        indicators = 9
        edges = edge_index
        features = None
        features = data
        features = features.view(indicators, -1, self.len_input)
        features = features.unsqueeze(2)
        need_concat = []
        res_att = 0
        for block in self.BlockList:
            features, res_att, spatial_gcn = block(features, res_att, edges)
            need_concat.append(features)
        final_x = torch.cat(need_concat, dim=-1)
        output1 = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        output = output1.view(-1, self.num_for_predict)
        return output


def make_model(adj, node_num, num_for_predict, adj_strg, feature_size):
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_of_d = 1
    nb_block = 2  # 4
    in_channels = 1
    nb_chev_filter = 32
    nb_time_filter = 32
    time_strides = 1
    num_for_predict = num_for_predict
    len_input = feature_size
    num_of_vertices = node_num
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    model = PDU(DEVICE, num_of_d, nb_block, in_channels,
                nb_chev_filter, nb_time_filter, time_strides, num_for_predict,
                len_input, num_of_vertices, d_model, d_k, d_v, n_heads)
    return model
