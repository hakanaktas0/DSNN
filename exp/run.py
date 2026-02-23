#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm
import pickle as pk
from torch_geometric_signed_directed import node_class_split
from torch_geometric.utils import to_undirected
from scipy.sparse import coo_matrix

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch_geometric.transforms import RandomNodeSplit

from torch_geometric.data import Data
from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion
from models.disc_models import DiscreteDiagSheafDiffusionReal,DiscreteBundleSheafDiffusionReal , DiscreteGeneralSheafDiffusionReal
from utils.heterophilic import get_dataset, get_fixed_splits
from utils.edge_data import generate_dataset_3class, in_out_degree, link_prediction_evaluation
from torch_geometric.utils.undirected import to_undirected
from sklearn.metrics import roc_auc_score


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def in_out_degree(edge_index, size, weight=None):
    if weight is None:
        A = coo_matrix((np.ones(len(edge_index[0])), (edge_index[0], edge_index[1])), shape=(size, size),
                       dtype=np.float32).tocsr()
    else:
        A = coo_matrix((weight, (edge_index[0], edge_index[1])), shape=(size, size), dtype=np.float32).tocsr()

    out_degree = np.sum(np.abs(A), axis=0).T
    in_degree = np.sum(np.abs(A), axis=1)
    degree = torch.from_numpy(np.c_[in_degree, out_degree]).float()
    return degree


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask].long())
    loss = nll
    loss.backward()

    optimizer.step()
    del out


def train_edge(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.train_index)
    nll = F.nll_loss(out, data.y_train.long())
    loss = nll
    loss.backward()

    optimizer.step()
    del out


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        probs = F.softmax(logits, dim=1)
        roc_auc_scores = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask].long())

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())

            if args.dataset in ["questions", 'questions_directed']:
                roc_auc = roc_auc_score(y_true=data.y[mask].cpu().numpy(), y_score=probs[mask].cpu().numpy()[:, 1])
                roc_auc_scores.append(roc_auc)
        if roc_auc_scores == []:
            roc_auc_scores = [0, 0, 0]
        return accs, preds, losses, roc_auc_scores


def acc(pred, label):
    # print(pred.shape, label.shape)
    correct = pred.eq(label).sum().item()
    acc = correct / len(pred)
    return acc


def test_edge(model, data):
    model.eval()
    accs, losses, preds = [], [], []
    with torch.no_grad():
        for index, y in [(data.train_index, data.y_train), (data.val_index, data.y_val),
                         (data.test_index, data.y_test)]:
            logits = model(data.x, index)
            pred_label = logits.max(dim=1)[1]
            accu = acc(pred_label, y)
            loss = F.nll_loss(logits, y.long())
            accs.append(accu)
            preds.append(pred_label.detach().cpu())
            losses.append(loss.detach().cpu())
        return accs, preds, losses


def test_edge_full(model, data):
    model.eval()
    accs, losses, preds = [], [], []
    full_accs = []

    with torch.no_grad():
        train_logits = model(data.x, data.train_index)
        val_logits = model(data.x, data.val_index)
        test_logits = model(data.x, data.test_index)
        train_loss = F.nll_loss(train_logits, data.y_train.long())
        val_loss = F.nll_loss(val_logits, data.y_val.long())
        test_loss = F.nll_loss(test_logits, data.y_test.long())

        [[val_acc_full, val_acc, val_auc, val_f1_micro, val_f1_macro],
         [test_acc_full, test_acc, test_auc,
          test_f1_micro, test_f1_macro]] = link_prediction_evaluation(val_logits, test_logits, data.y_val, data.y_test)

        [[train_acc_full, train_acc, train_auc, train_f1_micro, train_f1_macro],
         [test_acc_full, test_acc, test_auc,
          test_f1_micro, test_f1_macro]] = link_prediction_evaluation(train_logits, test_logits, data.y_train,
                                                                      data.y_test)
        accs.append(train_acc)
        accs.append(val_acc)
        accs.append(test_acc)
        preds.append(train_logits.detach().cpu())
        preds.append(val_logits.detach().cpu())
        preds.append(test_logits.detach().cpu())
        losses.append(train_loss.detach().cpu())
        losses.append(val_loss.detach().cpu())
        losses.append(test_loss.detach().cpu())
        full_accs.append(train_acc_full)
        full_accs.append(val_acc_full)
        full_accs.append(test_acc_full)

    return accs, preds, losses, full_accs


def get_data_split(data, split_number, n_splits=10):
    # TODO: Remove the Transpose operation
    train_mask = data["train_mask"][:, split_number % n_splits]
    val_mask = data["val_mask"][:, split_number % n_splits]
    test_mask = data["test_mask"][:, split_number % n_splits]
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)
    return data


def run_exp(args, dataset, model_cls, fold):
    # data = dataset[0]
    data = dataset[0] if (
                not args.synthetic and args.dataset not in ['roman_empire_directed', 'questions_directed', 'questions',
                                                            'chameleon_filtered',
                                                            'squirrel_filtered', 'snap-patents', 'arxiv-year',
                                                            'ogbn-arxiv','amazon_ratings_directed']) else dataset.clone()
    if args.dataset in ['roman_empire_directed', 'questions_directed', 'questions', 'chameleon_filtered',
                        'squirrel_filtered','amazon_ratings_directed','telegram'] or args.synthetic:
        data = get_data_split(data, fold)
    elif args.dataset in ['snap-patents', 'arxiv-year']:
        data = get_data_split(data, fold, n_splits=5)
    elif args.dataset in ['ogbn-arxiv']:
        data = data
    elif args.dataset in ['citeseer']:
        num_nodes = data.x.size(0)
        num_classes = int(data.y.max().item()) + 1
        # Apply RandomNodeSplit with specified parameters
        num_train_per_class = int(0.48 * num_nodes / num_classes)
        split_transform = RandomNodeSplit(
            split='random',
            num_splits=10,
            num_train_per_class=num_train_per_class,
            num_val=0.32,  # 32% for validation
            num_test=0.20  # 20% for testing
        )
        data = split_transform(data)
        data = get_data_split(split_transform(data), fold)
    else:
        data = get_fixed_splits(data, args.dataset, fold)

    # data = get_fixed_splits(data, args.dataset, fold)
    data = data.to(args.device)

    edge_index = to_undirected(data.edge_index)

    model = model_cls(edge_index, vars(args), directed_edge_index=data.edge_index, )
    model = model.to(args.device)

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args.sheaf_decay},
        {'params': other_params, 'weight_decay': args.weight_decay}
    ], lr=args.lr)

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_roc_auc = test_roc_auc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0
    for epoch in range(args.epochs):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss], [train_roc_auc, val_roc_auc, temp_test_roc_auc] = test(model, data)
        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
            }
            if args.dataset in ["questions", 'questions_directed']:
                res_dict[f'fold{fold}_train_roc_auc'] = train_roc_auc
                res_dict[f'fold{fold}_val_roc_auc'] = val_roc_auc
                res_dict[f'fold{fold}_test_roc_auc'] = temp_test_roc_auc
            wandb.log(res_dict, step=epoch)

        new_best_trigger = val_acc > best_val_acc if args.stop_strategy == 'acc' and args.output_dim != 2 else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_roc_auc = val_roc_auc
            test_roc_auc = temp_test_roc_auc
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.early_stopping:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")
    if args.dataset in ["questions", 'questions_directed']:
        print(f"Test roc auc: {test_roc_auc:.4f}")
        print(f"Best val roc auc: {best_val_roc_auc:.4f}")

    if "ODE" not in args.model:
        # Debugging for discrete models
        for i in range(len(model.sheaf_learners)):
            L_max = model.sheaf_learners[i].L.detach().max().item()
            L_min = model.sheaf_learners[i].L.detach().min().item()
            L_avg = model.sheaf_learners[i].L.detach().mean().item()
            L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
            print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

        with np.printoptions(precision=3, suppress=True):
            for i in range(0, args.layers):
                print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")
    if args.dataset in ["questions", 'questions_directed']:
        wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_test_roc_auc': test_roc_auc,
                   'best_val_roc_auc': best_val_roc_auc, 'best_epoch': best_epoch})
    else:
        wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args.min_acc else True

    return test_acc, best_val_acc, keep_running, best_val_roc_auc, test_roc_auc


def run_exp_edge(args, datasets, model_cls, fold):
    edges = datasets[fold]['graph']
    x = in_out_degree(edges, size).to(args.device)

    edge_data = Data(x=x, edge_index=edges)

    y_train = datasets[fold]['train']['label']
    y_val = datasets[fold]['validate']['label']
    y_test = datasets[fold]['test']['label']
    edge_data.y_train = torch.from_numpy(y_train).long().to(args.device)
    edge_data.y_val = torch.from_numpy(y_val).long().to(args.device)
    edge_data.y_test = torch.from_numpy(y_test).long().to(args.device)

    edge_data.train_index = torch.from_numpy(datasets[fold]['train']['pairs']).to(args.device)
    edge_data.val_index = torch.from_numpy(datasets[fold]['validate']['pairs']).to(args.device)
    edge_data.test_index = torch.from_numpy(datasets[fold]['test']['pairs']).to(args.device)

    # data = data.to(args.device)
    # edge_index = to_undirected(data.edge_index)
    # model = model_cls(edge_index, vars(args),directed_edge_index=data.edge_index,)
    # TODO USE THE EDGES FROM THE DATASETS GRAPH / MODEL TAKES BOTH DIRECTED AND UNDIRECTED EDGES
    model = model_cls(to_undirected(edges).to(args.device), vars(args), directed_edge_index=edges.to(args.device), )
    model = model.to(args.device)

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args.sheaf_decay},
        {'params': other_params, 'weight_decay': args.weight_decay}
    ], lr=args.lr)

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')

    best_val_acc_full = test_acc_full = 0
    best_val_loss_full = float('inf')

    val_loss_history = []
    val_acc_history = []

    val_loss_history_full = []
    val_acc_history_full = []

    best_epoch = 0

    best_epoch_full = 0

    bad_counter = 0
    for epoch in range(args.epochs):
        train_edge(model, optimizer, edge_data)
        if args.task != 2:
            [train_acc, val_acc, tmp_test_acc], preds, [
                train_loss, val_loss, tmp_test_loss] = test_edge(model, edge_data)
            train_acc_full, val_acc_full, tmp_test_acc_full = 0, 0, 0
        else:
            [train_acc, val_acc, tmp_test_acc], preds, [
                train_loss, val_loss, tmp_test_loss], [train_acc_full, val_acc_full,
                                                       tmp_test_acc_full] = test_edge_full(model, edge_data)
        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
            }
            if args.task == 2:
                res_dict[f'fold{fold}_train_acc_full'] = train_acc_full
                res_dict[f'fold{fold}_val_acc_full'] = val_acc_full
                res_dict[f'fold{fold}_tmp_test_acc_full'] = tmp_test_acc_full
            wandb.log(res_dict, step=epoch)

        # TODO implemetation for 3-class is missing (the acc_full values are 3-class accuracy values (I think))
        new_best_trigger = val_acc > best_val_acc if args.stop_strategy == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
            best_val_acc_full = val_acc_full
            # best_val_loss = val_loss
            test_acc_full = tmp_test_acc_full
            best_epoch_full = epoch
        else:
            bad_counter += 1

        new_best_trigger_full = val_acc_full > best_val_acc_full if args.stop_strategy == 'acc' else val_loss < best_val_loss
        if new_best_trigger_full:
            best_val_acc_full = val_acc_full
            # best_val_loss = val_loss
            test_acc_full = tmp_test_acc_full
            best_epoch_full = epoch
            bad_counter = 0

        if bad_counter == args.early_stopping:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch} | Best epoch full: {best_epoch_full}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Test acc full: {test_acc_full:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Best val acc full: {best_val_acc_full:.4f}")

    if "ODE" not in args.model:
        # Debugging for discrete models
        for i in range(len(model.sheaf_learners)):
            L_max = model.sheaf_learners[i].L.detach().max().item()
            L_min = model.sheaf_learners[i].L.detach().min().item()
            L_avg = model.sheaf_learners[i].L.detach().mean().item()
            L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
            print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

        with np.printoptions(precision=3, suppress=True):
            for i in range(0, args.layers):
                print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")

    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch,
               "best_test_acc_full": test_acc_full, "best_val_acc_full": best_val_acc_full})
    keep_running = False if test_acc < args.min_acc else True
    del edge_data
    return test_acc, best_val_acc, test_acc_full, best_val_acc_full, keep_running


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.head.object.hexsha


    if args.model == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args.model == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    elif args.model == 'DiagSheafReal':
        model_cls = DiscreteDiagSheafDiffusionReal
    elif args.model == 'BundleSheafReal':
        model_cls = DiscreteBundleSheafDiffusionReal
    elif args.model == 'GeneralSheafReal':
        model_cls = DiscreteGeneralSheafDiffusionReal
    else:
        raise ValueError(f'Unknown model {args.model}')
    if args.synthetic:
        dataset_path = os.path.join('synthetic_dataset', f'{args.dataset}.pk')
        with open(dataset_path, 'rb') as f:
            dataset = pk.load(f)
        directed_edges = dataset.edge_index
        dataset.edge_index = to_undirected(dataset.edge_index)
        dataset = node_class_split(dataset, train_size_per_class=0.6, val_size_per_class=0.2)
        dataset.x = in_out_degree(dataset.edge_index, dataset.y.size(-1))
        dataset.edge_index = directed_edges
        dataset.num_features = dataset.x.size(1)
        dataset.num_classes = dataset.y.max().item() + 1
    else:
        dataset = get_dataset(args.dataset)
    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # Add extra arguments
    # args.sha = sha
    args.graph_size = dataset[0].x.size(0) if (
                not args.synthetic and args.dataset not in ['roman_empire_directed', 'questions_directed', 'questions',
                                                            'chameleon_filtered',
                                                            'squirrel_filtered', 'arxiv-year', 'ogbn-arxiv',
                                                            'snap-patents','amazon_ratings_directed']) else dataset.x.size(
        0)  # type: ignore

    # Data(x=[183, 1703], edge_index=[2, 309], y=[183], train_mask=[183], val_mask=[183], test_mask=[183])
    # args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    class_3_results = []
    print(f"Running with wandb account: {args.entity}")
    print(args)
    # wandb.init(project="dsnn", config=vars(args), entity=args.entity)
    wandb.init(settings=wandb.Settings(mode="disabled", program="test.py", program_relpath="test.py"),
               project='dsnn')
    if  "arxiv" in args.dataset and len(dataset.y.shape) > 1:
        dataset.y = dataset.y.squeeze(-1)

    if args.pred_task == 'node_classification':
        for fold in tqdm(range(args.folds)):
            test_acc, best_val_acc, keep_running, best_val_roc_auc, test_roc_auc = run_exp(args, dataset, model_cls,
                                                                                           fold)
            if args.dataset in ["questions", 'questions_directed']:
                results.append([test_acc, best_val_acc, test_roc_auc, best_val_roc_auc])
            else:
                results.append([test_acc, best_val_acc])
            if not keep_running:
                break
    else:
        # data = dataset[0]
        # we just need the edge_index
        data = dataset[0] if (not args.synthetic and args.dataset not in ['roman_empire_directed', 'questions_directed',
                                                                          'questions', 'chameleon_filtered',
                                                                          'squirrel_filtered','amazon_ratings_directed']) else dataset.clone()
        if args.dataset in ['roman_empire_directed', 'questions_directed', 'questions', 'chameleon_filtered',
                            'squirrel_filtered','amazon_ratings_directed','telegram'] or args.synthetic:
            data = get_data_split(data, 0)
        elif args.dataset in ['citeseer']:
            num_nodes = data.x.size(0)
            num_classes = int(data.y.max().item()) + 1
            # Apply RandomNodeSplit with specified parameters
            num_train_per_class = int(0.48 * num_nodes / num_classes)
            split_transform = RandomNodeSplit(
                split='random',
                num_splits=10,
                num_train_per_class=num_train_per_class,
                num_val=0.32,  # 32% for validation
                num_test=0.20  # 20% for testing
            )
            data = split_transform(data)
            data = get_data_split(split_transform(data), 0)
        else:
            data = get_fixed_splits(data, args.dataset, 0)

        # data = get_fixed_splits(data, args.dataset, fold)
        size = torch.max(data.edge_index).item() + 1
        # TODO data is created each time.
        datasets = generate_dataset_3class(data.edge_index, size, splits=args.folds, task=args.task,
                                           label_dim=args.num_class_link)
        for fold in tqdm(range(args.folds)):
            test_acc, best_val_acc, test_acc_full, best_val_acc_full, keep_running = run_exp_edge(args, datasets,
                                                                                                  model_cls, fold)
            results.append([test_acc, best_val_acc])
            class_3_results.append([test_acc_full, best_val_acc_full])
            if not keep_running:
                break

    if args.pred_task == 'node_classification':
        if args.dataset in ["questions", 'questions_directed']:
            test_acc_mean, val_acc_mean, test_roc_auc_mean, val_roc_auc_mean = np.mean(results, axis=0) * 100
            test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
            test_roc_auc_std = np.sqrt(np.var(results, axis=0)[2]) * 100
            wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std,
                             'test_roc_auc': test_roc_auc_mean, 'val_roc_auc': val_roc_auc_mean,
                             'test_roc_auc_std': test_roc_auc_std, }
        else:
            test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
            test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100
            wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std, }
    else:  # TODO
        test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
        test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

        test_acc_mean_full, val_acc_mean_full = np.mean(class_3_results, axis=0) * 100
        test_acc_std_full = np.sqrt(np.var(class_3_results, axis=0)[0]) * 100

        wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std,
                         "test_acc_full": test_acc_mean_full, 'val_acc_full': val_acc_mean_full,
                         "test_acc_std_full": test_acc_std_full}

    # else:
    #
    wandb.log(wandb_results)
    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} | ')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
    if args.dataset in ["questions", 'questions_directed']:
        print(f'Test roc_auc: {test_roc_auc_mean:.4f} +/- {test_roc_auc_std:.4f} | Val acc: {val_roc_auc_mean:.4f}')
    if args.pred_task != 'node_classification':
        print(f'Test acc full: {test_acc_mean_full:.4f} +/- {test_acc_std_full:.4f} | Val acc: {val_acc_mean_full:.4f}')

