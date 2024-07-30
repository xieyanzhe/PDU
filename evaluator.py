import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
def evaluate(prediction, ground_truth, mask, epoch=None, unit=None, market='hope', report=None):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask) ** 2 \
                         / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0
    cash_flow = []
    realndcg5 = 0
    sharpe_li5 = []
    sharpe5_max = []
    sharpe_li10 = []
    sharpe_li1 = []
    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])

        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])

        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)

        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt

        realndcg5 += ndcg_score(np.array(list(gt_top5)).reshape(1, -1), np.array(list(pre_top5)).reshape(1, -1))
        performance['ndcg_score_top5'] = ndcg_score(np.array(list(gt_top5)).reshape(1, -1),
                                                    np.array(list(pre_top5)).reshape(1,
                                                                                     -1))
        performance['ndcg_score_top10'] = ndcg_score(np.array(list(gt_top10)).reshape(1, -1),
                                                     np.array(list(pre_top10)).reshape(1,
                                                                                       -1))

        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top
        sharpe_li1.append(real_ret_rat_top)
        real_ret_rat_top5 = 0
        real_max_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        for top in gt_top5:
            real_max_rat_top5 += ground_truth[top][i]
        real_ret_rat_top5 /= 5
        real_max_rat_top5 /= 5
        sharpe5_max.append(real_max_rat_top5)
        bt_long5 += real_ret_rat_top5
        sharpe_li5.append(real_ret_rat_top5)
        real_ret_rat_top10 = 0
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10
        sharpe_li10.append(real_ret_rat_top10)
    performance['sharpe5_max'] = (np.mean(sharpe5_max) / np.std(sharpe5_max)) * 15.87
    performance['realndcg5'] = realndcg5 / prediction.shape[1]
    performance['sharpe10'] = (np.mean(sharpe_li10) / np.std(sharpe_li10)) * 15.87
    performance['sharpe1'] = (np.mean(sharpe_li1) / np.std(sharpe_li1)) * 15.87
    performance['data5'] = sharpe_li5
    performance['data1'] = gt_top5
    return performance
