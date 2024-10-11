import torch.nn as nn
import torch.nn.functional as F
import torch
from PDUArch import make_model, Predictor
from utils import byol_loss_fn, transpose_node_to_graph
class MLP(nn.Module):
    def __init__(self, hid, sigma=False):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hid, 18),
            nn.LeakyReLU(),
            nn.Linear(18, 1),
        )

    def forward(self, x):
        return self.layers(x)
def add_small_random_gaussian_noise(input_data, mean=0, std=0.2, indicators=None):
    noise_list = []
    noise_shape = (input_data.shape[0] // indicators, input_data.shape[1])
    for i in range(indicators):
        noise = torch.normal(mean=mean, std=std, size=noise_shape)
        noise_list.append(noise)
    noise = torch.cat(noise_list, dim=0).to(input_data.device)
    noisy_data = input_data + noise
    return noisy_data


def random_mask(input_data, mask_prob, mask=None, indicators=None):
    if mask is not None:
        mask = mask
    else:

        mask_list = []
        mask_shape = (input_data.shape[0] // indicators, input_data.shape[1])
        for i in range(indicators):
            mask = (torch.rand(mask_shape) > mask_prob).float()
            mask_list.append(mask)
        mask = torch.cat(mask_list, dim=0).to(input_data.device)

    masked_data = input_data * mask
    return masked_data, mask


number = 1026


class PDUModel(nn.Module):
    def update_moving_average(self, beta=0.99):
        with torch.no_grad():
            for online_params, target_params in zip(self.backbone_online.parameters(),
                                                    self.backbone_target.parameters()):
                target_params.data = beta * target_params.data + (1.0 - beta) * online_params.data

    def __init__(self, in_feature=None, out_feature=None, node_num=number, revin=False, adj=None, adj_strg=None,
                 hidden_feature=None):
        super(PDUModel, self).__init__()
        self.device = 'cuda'
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.node_num = node_num
        self.revin = revin
        self.hidden_feature = hidden_feature

        self.predictor = Predictor(self.hidden_feature, out_feature, node_num,
                                   hidden_feature=self.hidden_feature, headers=3, block_num=2)
        self.backbone_online = make_model(adj=adj, node_num=node_num,
                                          num_for_predict=self.hidden_feature, adj_strg=adj_strg,
                                          feature_size=in_feature)
        self.backbone_target = make_model(adj=adj, node_num=node_num,
                                          num_for_predict=self.hidden_feature, adj_strg=adj_strg,
                                          feature_size=in_feature)
        self.dense_online = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_feature, 2 * self.hidden_feature),
            torch.nn.BatchNorm1d(2 * self.hidden_feature),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * self.hidden_feature, self.hidden_feature),
        )
        self.criterion = byol_loss_fn
        self.MLP = MLP(9, sigma=True)

    def forward(self, inputs, Ms=None, history_gt_batch=None, h_t=None, pyg_graph=None, type=None, trend_x=None,
                e=None):
        self.edges = pyg_graph.cuda()
        number = inputs.shape[0]
        data = inputs.view(number * 9, 4)
        trend_x = data.detach().clone()
        trend_x = add_small_random_gaussian_noise(trend_x, std=0.001, indicators=9)
        trend_x, mask = random_mask(trend_x, 0.001, indicators=9)
        if type == 'vel':
            h = self.backbone_online(data, self.edges, mask=type, e=e)
            pre = self.predictor(h, self.edges)
            pre = pre.view(number, 9)
            pre = self.MLP(pre)
            return pre, torch.tensor(0)
        features_x = self.backbone_online(data, self.edges, mask=type, e=e)
        features_t = self.backbone_online(trend_x, self.edges, mask=type, e=e)
        output = self.predictor(features_x, self.edges)
        x_online = self.dense_online(features_x)
        t_online = self.dense_online(features_t)
        with torch.no_grad():
            x_target = self.backbone_target(data, self.edges, mask=type, e=e)
            t_target = self.backbone_target(trend_x, self.edges, mask=type, e=e)
        x_online = transpose_node_to_graph(x_online, 9, self.hidden_feature)
        t_online = transpose_node_to_graph(t_online, 9, self.hidden_feature)
        x_target = transpose_node_to_graph(x_target, 9, self.hidden_feature)
        t_target = transpose_node_to_graph(t_target, 9, self.hidden_feature)
        loss = (self.criterion(F.normalize(x_online), F.normalize(t_target.detach())) +
                self.criterion(F.normalize(t_online), F.normalize(x_target.detach())))
        loss = loss.mean()
        output = output.view(number, 9)
        output = self.MLP(output)
        return output, loss
