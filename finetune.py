import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
lo
from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from searchspace import *
from sklearn.metrics import roc_auc_score, mean_squared_error

from splitters import scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

import sys
from log import *
import warnings
warnings.filterwarnings('ignore')
import traceback


# criterion = nn.BCEWithLogitsLoss(reduction = "none")


# Training settings
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--ft_mode', type=str, default='fully_latest', help='how to adapt for downstream tasks')
parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--lr_scale', type=float, default=1, help='relative learning rate for the feature extraction layer (default: 1)')
parser.add_argument('--decay', type=float, default=0, help='weight decay (default: 0)')
parser.add_argument('--max_norm', type=float, default=1, help='gradient clipping')
parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5).')
parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')
parser.add_argument('--adapter_dim', type=int, default=2, help='adapter dimensions (default: 8)')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
parser.add_argument('--disable_search', action='store_true', help='whether to disable search or not')
parser.add_argument('--disable_pretrain', action='store_true', help='whether to disable pre-train or not')

# args.graph_pooling and args.JK to be refined in model.py and convs.py later
# parser.add_argument('--graph_pooling', type=str, default="mean", help='graph level pooling (sum, mean, max, set2set, attention)')
# parser.add_argument('--JK', type=str, default="last", help='how the node features across layers are combined. last, sum, max or concat')

parser.add_argument('--gnn_type', type=str, default="gin")
parser.add_argument('--dataset', type=str, default='sider', help='root directory of dataset.')
parser.add_argument('--task_type', type=str, default='classification', help="Task type.")
parser.add_argument('--input_model_file', type=str, default='model_gin/graphmae.pth', help='filename to read the model (if there is any)')
parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
parser.add_argument('--num_runs', type=int, default=15, help="Number of runs per command.")
parser.add_argument('--runseed_min', type=int, default=0, help="Seed for minibatch selection, random initialization.")
parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
parser.add_argument('--temp', type=float, default=0.1, help='temperature of softmax')
parser.add_argument('--log_dir', type=str, default='./log_latest', help='log directory')
parser.add_argument('--backbone', type=str, default='', help='name of gnn backbone, to be set in set_up_log()')
parser.add_argument('--obj', type=str, default='', help='name of self-supervised objective, to be set in set_up_log()')
parser.add_argument('--pretrained_model', type=str, default='', help='name of pretrained_model, to be set in set_up_log()')
parser.add_argument('--file_path', type=str, default='', help='log path, to be set in set_up_log()')
parser.add_argument('--summary_file', type=str, default='summary.log', help='results summary')
args = parser.parse_args()
sys_argv = sys.argv


def check_search(args):
    if len(ADAPTER_CANDIDATES) * len(JK_CANDIDATES) * len(POOL_CANDIDATES) == 1 or args.disable_search == True:
        search_flag, search_adapter, search_jk, search_pool = False, False, False, False
    else:
        search_flag = True
        search_adapter = len(ADAPTER_CANDIDATES) > 1
        search_jk = len(JK_CANDIDATES) > 1
        search_pool = len(POOL_CANDIDATES) > 1
    return search_flag, search_adapter, search_jk, search_pool


# def freeze_parameters(args, model):
#     assert args.ft_mode in ['fully', 'decoder_only',
#                             'last_1', 'last_2', 'last_3', 'last_4',
#                             # 'adapter_se', 'adapter_pa',
#                             'fully_adapter_auto', 'decoder_only_adapter_auto']
#     if args.ft_mode == 'fully':
#         pass
#     elif 'adapter' in args.ft_mode or args.ft_mode == 'decoder_only':
#         for n, p in model.named_parameters():
#             if 'gnn' in n:
#                 p.requires_grad = False
#     # !! currently the following last_k implementation is only for GIN backbone
#     elif 'last' in args.ft_mode:
#         tunable_layer = int(args.ft_mode.split('_')[-1])
#         frozen_layer = args.num_layer - tunable_layer
#         frozen_p_n = []
#         for k in range(frozen_layer):
#             frozen_p_n.extend([f'gnn.gnns.{k}.mlp.0.weight',
#                                f'gnn.gnns.{k}.mlp.0.bias',
#                                f'gnn.gnns.{k}.mlp.2.weight',
#                                f'gnn.gnns.{k}.mlp.2.bias',
#                                f'gnn.gnns.{k}.edge_embedding1.weight',
#                                f'gnn.gnns.{k}.edge_embedding2.weight',
#                                f'gnn.batch_norms.{k}.weight',
#                                f'gnn.batch_norms.{k}.bias'])
#         for n, p in model.named_parameters():
#             if n in frozen_p_n:
#                 p.requires_grad = False
#     else:
#         raise NotImplementedError


def freeze_parameters(args, model):
    assert args.ft_mode in ['fully', 'fully_skip', 'fully_latest',
                            'decoder_only',
                            'last_1', 'last_2', 'last_3', 'last_4',
                            'adapter_pa', 'adapter_se', # new baselines: re-run without architecture change
                            # 'adapter_se_manual', 'adapter_se_auto', # se; se/none search (freeze ginconvs)
                            # 'adapter_pa_manual', 'adapter_pa_auto' # pa; pa/none search (freeze ginconvs)
                            'fully_adapter_pa', 'fully_adapter_se',
                            'fully_auto_adapter_pa', 'fully_auto_adapter_se',
                            # 'adapter_pa', 'adapter_se',
                            'auto_adapter_pa', 'auto_adapter_se'
                            ]

    if args.ft_mode in ['fully_latest', # 3 dimensions
                        'fully', # 2 dimensions,
                        'fully_skip', # 3 dimensions (+ pure auto skip connection)
                        'fully_adapter_pa', 'fully_adapter_se', # 3 dimensions
                        'fully_auto_adapter_pa', 'fully_auto_adapter_se' # 3 dimensions
                        ]:
        pass

    elif args.ft_mode in ['decoder_only',
                          'adapter_pa', 'adapter_se']:
        for n, p in model.named_parameters():
            if 'gnn' in n:
                p.requires_grad = False

    # elif args.ft_mode in ['fully_adapter_pa', 'fully_adapter_se',
    #                       'fully_auto_adapter_pa', 'fully_auto_adapter_se']:
    #     pass

    # elif args.ft_mode in ['adapter_pa', 'adapter_se',
    #                       'auto_adapter_pa', 'auto_adapter_se']:
    #     for n, p in model.named_parameters():
    #         if 'gnn' in n and 'batch_norms' not in n:
    #             p.requires_grad = False

    # !! currently the following last_k implementation is only for GIN backbone
    elif 'last' in args.ft_mode:
        tunable_layer = int(args.ft_mode.split('_')[-1])
        frozen_layer = args.num_layer - tunable_layer
        frozen_p_n = []
        for k in range(frozen_layer):
            frozen_p_n.extend([f'gnn.gnns.{k}.mlp.0.weight',
                               f'gnn.gnns.{k}.mlp.0.bias',
                               f'gnn.gnns.{k}.mlp.2.weight',
                               f'gnn.gnns.{k}.mlp.2.bias',
                               f'gnn.gnns.{k}.edge_embedding1.weight',
                               f'gnn.gnns.{k}.edge_embedding2.weight',
                               f'gnn.batch_norms.{k}.weight',
                               f'gnn.batch_norms.{k}.bias'])
        for n, p in model.named_parameters():
            if n in frozen_p_n:
                p.requires_grad = False

    else:
        raise NotImplementedError


def train(args, model, device, loader, optimizer):
    if args.task_type == "classification":
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    else:
        criterion = nn.MSELoss(reduction='none')

    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        # print(total_norm)
        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    if args.task_type == "classification":
        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                is_valid = y_true[:,i]**2 > 0
                # sometimes error occurs (maybe due to the package version??)
                roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))
        return sum(roc_list)/len(roc_list) #y_true.shape[1]

    else:
        rmse = mean_squared_error(y_true, y_scores, squared=False)
        return rmse


def run(model, train_loader, val_loader, test_loader, device, logger, args):
    # set up optimizer
    # different learning rate for different part of GNN
    # model_param_group = []
    # model_param_group.append({"params": model.gnn.parameters()})
    # if args.graph_pooling == "attention":
    #     model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    # model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    # model_param_group.append({"params": [model.jk_alphas, model.pool_alphas], "lr":args.lr*args.lr_scale})
    # optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # print(optimizer)

    train_auc_list = []
    val_auc_list = []
    test_auc_list = []


    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_auc = eval(args, model, device, train_loader)
        else:
            print("omit the training auc computation")
            train_auc = 0
        val_auc = eval(args, model, device, val_loader)
        test_auc = eval(args, model, device, test_loader)

        logger.info(f"[{model.mode}] Epoch: {epoch} === train: {np.round(train_auc, 4)} === val: {np.round(val_auc, 4)} === test: {np.round(test_auc, 4)}")
        val_auc_list.append(val_auc)
        test_auc_list.append(test_auc)
        train_auc_list.append(train_auc)

        # print(model.adapter_alphas)
        # print(model.adapter_ws)
        # print(model.jk_alphas)
        # print(model.jk_ws)

    return train_auc_list, val_auc_list, test_auc_list


def main():
    # if args.runseed in [0, 2]:
    #     print('a' + 1)
    # try:
    logger = set_up_log(args, sys_argv)
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset in ["esol", "freesolv", "lipophilicity"]:
        num_tasks = 1
        args.task_type = "regression"
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    logger.info(dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        logger.info("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        logger.info("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        logger.info("random scaffold")
    else:
        raise ValueError("Invalid split option.")
    logger.info(train_dataset[0])

    drop_last = False if args.dataset != 'freesolv' else True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=drop_last)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers, drop_last=drop_last)

    logger.info('**************************** search phase ****************************')
    t1 = time.time()

    # logger.info(f'Whether conduct search: {search_flag}')
    search_flag, search_adapter, search_jk, search_pool = check_search(args)
    logger.info(f'search_flag: {search_flag} === search_adapter: {search_adapter} search_jk: {search_jk} === search_pool: {search_pool}')

    # architectures are derived based on the last epoch
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, ft_mode=args.ft_mode, adapter_dim=args.adapter_dim, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type, temp = args.temp, mode = 'search')
    # if not args.input_model_file == "":
    if (args.input_model_file != "") and (args.disable_pretrain == False):
        model.from_pretrained(args.input_model_file)
    freeze_parameters(args, model)
    # for n, p in model.named_parameters():
    #     print(n)
    #     print(p.requires_grad)
    # return
    model.to(device)
    if search_flag == True:
        _, _, _ = \
            run(model, train_loader, val_loader, test_loader, device, logger, args)
    else:
        pass
    model.derive_arch()

    t2 = time.time()

    logger.info('**************************** re-train phase ****************************')
    # report best test w/wo val
    model_max = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, ft_mode=args.ft_mode, adapter_dim=args.adapter_dim, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type, temp = args.temp, mode = 're-train')
    # if not args.input_model_file == "":
    if (args.input_model_file != "") and (args.disable_pretrain == False):
        model_max.from_pretrained(args.input_model_file)
    model_max.from_searched(model)
    del model
    freeze_parameters(args, model_max)
    # for n, p in model_max.named_parameters():
    #     print(n)
    #     print(p.requires_grad)
    model_max.to(device)

    train_auc_list, val_auc_list, test_auc_list = \
        run(model_max, train_loader, val_loader, test_loader, device, logger, args)

    t3 = time.time()

    time_recorder = [np.round((t2 - t1)/3600, 2), np.round((t3 - t2)/3600, 2)]
    # time_recorder = [np.round((t2 - t1), 2), np.round((t3 - t2), 2)]
    # print(model_max.searched_arch['ADAPTER'])
    # print(len(model_max.searched_arch['ADAPTER']))
    # print(model_max.searched_arch['ADAPTER'][0])
    # return

    logger.info('*************************** summarize results **************************')
    model_max.search_flag, model_max.search_adapter, model_max.search_jk, model_max.search_pool = search_flag, search_adapter, search_jk, search_pool

    if args.task_type == 'classification':
        model_max.best_test_w_val_epoch = val_auc_list.index(max(val_auc_list))
        model_max.best_test_wo_val_epoch = test_auc_list.index(max(test_auc_list))
    else:
        model_max.best_test_w_val_epoch = val_auc_list.index(min(val_auc_list))
        model_max.best_test_wo_val_epoch = test_auc_list.index(min(test_auc_list))
    model_max.best_test_w_val = np.round(test_auc_list[model_max.best_test_w_val_epoch], 4)
    model_max.best_test_wo_val = np.round(test_auc_list[model_max.best_test_wo_val_epoch], 4)

    logger.info(f'runseed: {args.runseed}')
    logger.info(f'dataset: {args.dataset}')
    logger.info(f'best_test_w_val: {model_max.best_test_w_val}')
    logger.info(f'best_test_wo_val: {model_max.best_test_wo_val}')
    logger.info(f'w_val_epoch: {model_max.best_test_w_val_epoch}')
    logger.info(f'wo_val_epoch: {model_max.best_test_wo_val_epoch}')
    logger.info(f'backbone: {args.backbone}')
    logger.info(f'obj: {args.obj}')
    logger.info(f'temp: {args.temp}')
    logger.info(f'search_adapter: {model_max.search_adapter}')
    logger.info(f'search_jk: {model_max.search_jk}')
    logger.info(f'search_pool: {model_max.search_pool}')
    logger.info(f"adapter: {'--'.join(model_max.searched_arch['ADAPTER'])}")
    logger.info(f"jk: {model_max.searched_arch['JK']}")
    logger.info(f"pool: {model_max.searched_arch['POOL']}")
    logger.info(f"bs: {args.batch_size}")
    logger.info(f"ft_mode: {args.ft_mode}")
    logger.info(f"t_search (hour): {time_recorder[0]}")
    logger.info(f"t_retrain (hour): {time_recorder[1]}")
    # logger.info(f'jk_alphas: {list(map(lambda x: round(x, 2), model_max.jk_alphas.tolist()))}')
    # logger.info(f'pool_alphas: {list(map(lambda x: round(x, 2), model_max.pool_alphas.tolist()))}')
    logger.info(f'adapter_dim: {args.adapter_dim}')
    logger.info(f'disable_pretrain: {args.disable_pretrain}')

    save_performance_result(args, logger, model_max, time_recorder)

    # except Exception as e:
    #     logging.error(f"main Exception: {traceback.format_exc()}")
    #     raise


if __name__ == "__main__":
    cnt = 0 # success and fail runs
    num_runs = 0 # success runs
    while num_runs < args.num_runs:
        args.runseed = args.runseed_min + cnt
        cnt += 1
        num_runs += 1
        try:
            main()
        except Exception as e:
            num_runs -= 1
            logging.error(f"main Exception: {traceback.format_exc()}")
            continue
