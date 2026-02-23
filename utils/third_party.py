import numpy as np
import scipy
import os
import gdown

import torch
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as transforms
from torch_geometric.data import download_url


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask

# Taken verbatim from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data_utils.py#L39
def even_quantile_labels(vals, nclasses, verbose=True):
    """partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label


# adapting
# https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data_utils.py#L221
# load splits from here https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
def process_fixed_splits(splits_lst, num_nodes):
    n_splits = len(splits_lst)
    train_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    for i in range(n_splits):
        train_mask[splits_lst[i]["train"], i] = 1
        val_mask[splits_lst[i]["valid"], i] = 1
        test_mask[splits_lst[i]["test"], i] = 1
    return train_mask, val_mask, test_mask


# adapting - https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/dataset.py#L257
def load_snap_patents_mat(n_classes=5, root="datasets/"):
    dataset_drive_url = {"snap-patents": "1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia"}
    splits_drive_url = {"snap-patents": "12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N"}

    # Build dataset folder
    if not os.path.exists(f"{root}snap_patents"):
        os.mkdir(f"{root}snap_patents")

    # Download the data
    if not os.path.exists(f"{root}snap_patents/snap_patents.mat"):
        p = dataset_drive_url["snap-patents"]
        print(f"Snap patents url: {p}")
        gdown.download(
            id=dataset_drive_url["snap-patents"],
            output=f"{root}snap_patents/snap_patents.mat",
            quiet=False,
        )

    # Get data
    fulldata = scipy.io.loadmat(f"{root}snap_patents/snap_patents.mat")
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
    node_feat = torch.tensor(fulldata["node_feat"].todense(), dtype=torch.float)
    num_nodes = int(fulldata["num_nodes"])
    years = fulldata["years"].flatten()
    label = even_quantile_labels(years, n_classes, verbose=False)
    label = torch.tensor(label, dtype=torch.long)

    # Download splits
    name = "snap-patents"
    if not os.path.exists(f"{root}snap_patents/{name}-splits.npy"):
        assert name in splits_drive_url.keys()
        gdown.download(
            id=splits_drive_url[name],
            output=f"{root}snap_patents/{name}-splits.npy",
            quiet=False,
        )

    # Get splits
    splits_lst = np.load(f"{root}snap_patents/{name}-splits.npy", allow_pickle=True)
    train_mask, val_mask, test_mask = process_fixed_splits(splits_lst, num_nodes)
    data = Data(
        x=node_feat,
        edge_index=edge_index,
        y=label,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=n_classes,
    )

    return data

def get_arxiv_year_dataset():
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=transforms.ToSparseTensor(), root='datasets/')
    evaluator = Evaluator(name="ogbn-arxiv")
    y = even_quantile_labels(dataset.data.node_year.flatten().numpy(), nclasses=5, verbose=False)
    dataset.data.y = torch.as_tensor(y)
    # Tran, val and test masks are required during preprocessing. Setting them here to dummy values as
    # they are overwritten later for this dataset (see get_dataset_split function below)
    dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = 0, 0, 0
    # Create directory for this dataset
    os.makedirs(os.path.join('datasets/', 'arxiv_year', "raw"), exist_ok=True)
    data = dataset.data
    num_nodes = data["y"].shape[0]
    github_url = f"https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/"
    split_file_name = "arxiv-year-splits.npy"
    local_dir = os.path.join('datasets/', 'arxiv_year', "raw")

    download_url(os.path.join(github_url, split_file_name), local_dir, log=False)
    splits = np.load(os.path.join(local_dir, split_file_name), allow_pickle=True)
    dataset.data['train_mask'] = np.zeros((num_nodes,5))
    dataset.data['val_mask'] = np.zeros((num_nodes, 5))
    dataset.data['test_mask'] = np.zeros((num_nodes, 5))
    for i in range(5):
        dataset.data['train_mask'][:, i] = get_mask(splits[i]['train'],num_nodes)
        dataset.data['val_mask'][:, i] = get_mask(splits[i]['valid'],num_nodes)
        dataset.data['test_mask'][:, i] = get_mask(splits[i]['test'],num_nodes)
    dataset.data.num_classes = np.unique(dataset.data.y.numpy()).shape[0]
    return dataset.data


def get_ogbn_arxiv_dataset():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=transforms.ToSparseTensor(), root='datasets/')
    evaluator = Evaluator(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()
    dataset.data.train_mask = get_mask(split_idx["train"], dataset.data.num_nodes)
    dataset.data.val_mask = get_mask(split_idx["valid"], dataset.data.num_nodes)
    dataset.data.test_mask = get_mask(split_idx["test"], dataset.data.num_nodes)
    dataset.data.num_classes = np.unique(dataset.data.y.numpy()[:,0]).shape[0]
    return dataset.data

