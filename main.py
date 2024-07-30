import copy

import torch
import random
import argparse
import torch.nn as nn
import torch.nn.functional as F
import sys
import torch.optim as optim
import time
from tqdm import tqdm
from evaluator import evaluate
from load_data import *
import pickle

from PUD import PDUModel
from utils import DataGraph
from utils import get_graph_MS

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
device = 'cuda'


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)
def trr_loss_mse_rank(pred, post_cov, history_price_batch, history_gt_batch, base_price, ground_truth, future_price,
                      mask, alpha, no_stocks):
    return_ratio = torch.div((pred - base_price), base_price)
    reg_loss = weighted_mse_loss(return_ratio, ground_truth, mask)
    all_ones = torch.ones(no_stocks, 1).to(device)

    pre_pw_dif = (torch.matmul(return_ratio, torch.transpose(all_ones, 0, 1))
                  - torch.matmul(all_ones, torch.transpose(return_ratio, 0,
                                                           1)))
    gt_batch = ground_truth
    gt_pw_dif = (
            torch.matmul(all_ones, torch.transpose(gt_batch, 0, 1)) -
            torch.matmul(gt_batch, torch.transpose(all_ones, 0, 1))
    )
    mask_pw = torch.matmul(mask, torch.transpose(mask, 0, 1))
    rank_loss = torch.mean(
        F.relu(
            ((pre_pw_dif * gt_pw_dif) * mask_pw)))
    loss = rank_loss + reg_loss + 0.5 * post_cov
    del mask_pw, gt_pw_dif, pre_pw_dif, all_ones
    return loss, rank_loss, reg_loss, return_ratio


class Stock_PDU:
    def __init__(self, data_path, market_name, tickers_fname, n_node,
                 parameters, steps=1, epochs=None, early_stop_count=0, early_stop_n=3, indicators=None, flat=False,
                 gpu=True, in_pro=False, seed=0, weight=0.6, num_basis=15, args=None, hidden=None):
        self.hidden = hidden
        self.args = args
        self.seed = seed
        self.weight = weight
        self.num_basis = num_basis
        self.early_stop_count = early_stop_count
        self.early_stop_count = early_stop_n
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        self.train_data = pickle.load(open('../data/relation/NASDAQ_File.txt', 'rb'))
        self.n_node = n_node
        self.graph_data = DataGraph(self.train_data, len(self.tickers))
        self.pyg_graph = get_graph_MS(self.graph_data.graph)
        print('tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)
        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
        if indicators is None:
            self.indicators = len(self.tickers)
        else:
            self.indicators = indicators
        self.Stock_num = len(self.tickers)
        self.in_dim = 64
        self.emb_size = 64
        self.days = 4
        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5
        self.gpu = gpu

    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
            np.expand_dims(mask_batch, axis=1), \
            np.expand_dims(
                self.price_data[:, offset + seq_len - 1], axis=1
            ), \
            np.expand_dims(
                self.price_data[:, offset + seq_len], axis=1
            ), \
            np.expand_dims(
                self.price_data[:, offset:offset + seq_len], axis=1
            ), \
            np.expand_dims(
                self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
            ), \
            np.expand_dims(
                self.gt_data[:, offset + self.steps:offset + seq_len + self.steps], axis=1
            ), \
            np.expand_dims(
                self.gt_data[:, offset:offset + seq_len], axis=1
            )

    def train(self):
        global df
        model = PDUModel(in_feature=4, out_feature=1, node_num=1026, revin=False, adj=self.graph_data.graph,
                         adj_strg=None, hidden_feature=self.hidden).cuda()
        index = 0
        for p in model.parameters():
            index += 1
            if p.dim() > 1:
                nn.init.xavier_uniform_(
                    p)
            else:
                nn.init.uniform_(p)
        optimizer_hgat = optim.Adam(model.parameters(),
                                    lr=self.parameters['lr'],
                                    weight_decay=1e-3)
        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)

        get_hist = self.eod_data[:, :756, :]
        get_hist = torch.FloatTensor(get_hist).cuda()
        for i in range(self.epochs):
            print('epoch:', i)
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            model.train()
            for j in tqdm(range(self.valid_index - self.parameters['seq'] - self.steps + 1)):
                emb_batch, mask_batch, price_batch, future_price, history_price_batch, gt_batch, future_gt_batch, history_gt_batch = self.get_batch(
                    batch_offsets[j])
                optimizer_hgat.zero_grad()
                output, post_cov = model.forward(torch.FloatTensor(emb_batch).to(device),
                                                 torch.FloatTensor(self.graph_data.graph).cuda(),
                                                 history_gt_batch,
                                                 h_t=get_hist, pyg_graph=self.pyg_graph, type='train', e=j)
                cur_loss, cur_rank_loss, cur_reg_loss, curr_rr_train = trr_loss_mse_rank(output, post_cov,
                                                                                         torch.squeeze(
                                                                                             torch.FloatTensor(
                                                                                                 history_price_batch).to(
                                                                                                 device)),
                                                                                         torch.squeeze(
                                                                                             torch.FloatTensor(
                                                                                                 future_gt_batch).to(
                                                                                                 device)),
                                                                                         torch.FloatTensor(
                                                                                             price_batch).to(
                                                                                             device),
                                                                                         torch.FloatTensor(
                                                                                             gt_batch).to(
                                                                                             device),
                                                                                         torch.FloatTensor(
                                                                                             future_price).to(
                                                                                             device),
                                                                                         torch.FloatTensor(
                                                                                             mask_batch).to(
                                                                                             device),
                                                                                         self.parameters[
                                                                                             'alpha'],
                                                                                         self.indicators)
                all_loss = cur_loss
                all_loss.backward()
                optimizer_hgat.step()
                model.update_moving_average()
                tra_loss += all_loss.detach().cpu().item()
                tra_reg_loss += cur_reg_loss.detach().cpu().item()
                tra_rank_loss += cur_rank_loss.detach().cpu().item()
            print('Train Loss:',
                  tra_loss / (self.test_index - self.parameters['seq'] - self.steps + 1)
                  )

            with torch.no_grad():
                cur_valid_pred = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_gt = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                cur_valid_mask = np.zeros(
                    [len(self.tickers), self.test_index - self.valid_index],
                    dtype=float
                )
                val_loss = 0.0
                val_reg_loss = 0.0
                val_rank_loss = 0.0
                model.eval()
                for cur_offset in tqdm(range(
                        self.valid_index - self.parameters['seq'] - self.steps + 1,
                        self.test_index - self.parameters['seq'] - self.steps + 1
                )):
                    emb_batch, mask_batch, price_batch, future_price, history_price_batch, gt_batch, future_gt_batch, history_gt_batch = self.get_batch(
                        cur_offset)
                    output_val, post_cov = model(torch.FloatTensor(emb_batch).to(device),
                                                 torch.FloatTensor(self.graph_data.graph).cuda(),
                                                 history_gt_batch=history_gt_batch, h_t=get_hist,
                                                 pyg_graph=self.pyg_graph, type='vel', e=cur_offset)
                    cur_loss, cur_rank_loss, cur_reg_loss, cur_rr = trr_loss_mse_rank(output_val, post_cov,
                                                                                      torch.squeeze(
                                                                                          torch.FloatTensor(
                                                                                              history_price_batch).to(
                                                                                              device)),
                                                                                      torch.squeeze(
                                                                                          torch.FloatTensor(
                                                                                              future_gt_batch).to(
                                                                                              device)),
                                                                                      torch.FloatTensor(
                                                                                          price_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(
                                                                                          gt_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(
                                                                                          future_price).to(
                                                                                          device),
                                                                                      torch.FloatTensor(
                                                                                          mask_batch).to(
                                                                                          device),
                                                                                      self.parameters['alpha'],
                                                                                      self.indicators)
                    cur_rr = cur_rr.detach().cpu().numpy().reshape((len(self.tickers), 1))
                    val_loss += cur_loss.detach().cpu().item()
                    val_reg_loss += cur_reg_loss.detach().cpu().item()
                    val_rank_loss += cur_rank_loss.detach().cpu().item()
                    cur_valid_pred[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_valid_gt[:, cur_offset - (self.valid_index -
                                                  self.parameters['seq'] -
                                                  self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_valid_mask[:, cur_offset - (self.valid_index -
                                                    self.parameters['seq'] -
                                                    self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
                print('Valid LOSS:',
                      val_loss / (self.test_index - self.valid_index)
                      )
                cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt,
                                          cur_valid_mask, report=False)
                print('\t Valid preformance:sharpe5: ', cur_valid_perf['sharpe5'], 'ndcg_score_top5',
                      cur_valid_perf['ndcg_score_top5'], '  mrrt', cur_valid_perf['mrrt'], )
                cur_test_pred = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_gt = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                cur_test_mask = np.zeros(
                    [len(self.tickers), self.trade_dates - self.test_index],
                    dtype=float
                )
                test_loss = 0.0
                test_reg_loss = 0.0
                test_rank_loss = 0.0
                model.eval()
                for cur_offset in tqdm(
                        range(self.test_index - self.parameters['seq'] - self.steps + 1,
                              self.trade_dates - self.parameters['seq'] - self.steps + 1)
                ):
                    emb_batch, mask_batch, price_batch, future_price, history_price_batch, gt_batch, future_gt_batch, history_gt_batch = self.get_batch(
                        cur_offset)

                    output_test, post_cov = model(torch.FloatTensor(emb_batch).to(device),
                                                  torch.FloatTensor(self.graph_data.graph).cuda(),
                                                  history_gt_batch=history_gt_batch, h_t=get_hist,
                                                  pyg_graph=self.pyg_graph, type='test', e=cur_offset)
                    cur_loss, cur_rank_loss, cur_reg_loss, cur_rr = trr_loss_mse_rank(output_test, post_cov,
                                                                                      torch.squeeze(
                                                                                          torch.FloatTensor(
                                                                                              history_price_batch).to(
                                                                                              device)),
                                                                                      torch.squeeze(
                                                                                          torch.FloatTensor(
                                                                                              future_gt_batch).to(
                                                                                              device)),
                                                                                      torch.FloatTensor(
                                                                                          price_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(
                                                                                          gt_batch).to(
                                                                                          device),
                                                                                      torch.FloatTensor(
                                                                                          future_price).to(
                                                                                          device),
                                                                                      torch.FloatTensor(
                                                                                          mask_batch).to(
                                                                                          device),
                                                                                      self.parameters['alpha'],
                                                                                      self.indicators)
                    cur_rr = cur_rr.detach().cpu().numpy().reshape((len(self.tickers), 1))
                    test_loss += (cur_loss).detach().cpu().item()
                    test_reg_loss += cur_reg_loss.detach().cpu().item()
                    test_rank_loss += cur_rank_loss.detach().cpu().item()
                    cur_test_pred[:, cur_offset - (self.test_index -
                                                   self.parameters['seq'] -
                                                   self.steps + 1)] = \
                        copy.copy(cur_rr[:, 0])
                    cur_test_gt[:, cur_offset - (self.test_index -
                                                 self.parameters['seq'] -
                                                 self.steps + 1)] = \
                        copy.copy(gt_batch[:, 0])
                    cur_test_mask[:, cur_offset - (self.test_index -
                                                   self.parameters['seq'] -
                                                   self.steps + 1)] = \
                        copy.copy(mask_batch[:, 0])
            print('Test MSE LOSS:',
                  test_loss / (self.trade_dates - self.test_index)
                  )
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask, self.parameters['unit'], i,
                                     self.market_name, report=True)
            print('\t Test performance:', 'sharpe5:', cur_test_perf['sharpe5'], 'ndcg_score_top5:',
                  cur_test_perf['ndcg_score_top5'], 'mrrt', cur_test_perf['mrrt'], 'Test loss',
                  test_loss / (self.test_index - self.valid_index))
            np.set_printoptions(threshold=sys.maxsize)
    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01-1')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.01,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='sector_industry',
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    parser.add_argument('-node', default=1026, help='n_node')
    parser.add_argument('-seed', default=57, help='seed')
    args = parser.parse_args()
    args.gpu = (args.gpu == 1)
    args.inner_prod = (args.inner_prod == 1)
    market_name = 'NASDAQ'
    args.t = market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    parameters = {'seq': int(4), 'unit': int(args.u), 'lr': float(0.01),
                  'alpha': float(args.a)}
    PDU_NET = Stock_PDU(
        data_path=args.p,
        market_name=market_name,
        tickers_fname=args.t,
        n_node=args.node,
        parameters=parameters,
        steps=1, epochs=300,
        early_stop_count=0,
        early_stop_n=500,
        indicators=None, gpu=args.gpu,
        in_pro=args.inner_prod,
        seed=64, args=args,
        hidden=64
    )
    PDU_NET.train()
