from config import args
import joblib
import numpy as np
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import torch
import random
from tqdm import tqdm


class MyDataLoader(object):

    def __init__(self, dataset, batch_size, mini_batch_size=0):
        self.total = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        if mini_batch_size == 0:
            self.mini_batch_size = self.batch_size

    def __getitem__(self, item):
        ceil = (item + 1) * self.batch_size
        sub_dataset = self.dataset[ceil - self.batch_size:ceil]
        if ceil >= self.total:
            random.shuffle(self.dataset)
        return DataLoader(sub_dataset, batch_size=self.mini_batch_size)

    def __len__(self):
        if self.total == 0:
            return 0
        return (self.total - 1) // self.batch_size + 1


def split_train_valid_test(data, dataset, train_size, valid_part=0.1):
    train_data = []
    test_data = []
    with open(f"corpus_v1/{dataset}.split.txt", "r") as f:
        train_test = f.read().strip().split("\n")
    for tt, d in zip(train_test, data):
        if tt == 'test':
            test_data.append(d)
        elif tt == 'train':
            train_data.append(d)

    random.shuffle(train_data)
    valid_size = round(valid_part * train_size)
    valid_data = train_data[:valid_size]
    train_data = train_data[valid_size:]
    return train_data, test_data, valid_data


# old
# def split_train_valid_test(data, train_size, valid_part=0.1):
#     train_data = data[:train_size]
#     test_data = data[train_size:]
#     random.shuffle(train_data)
#     valid_size = round(valid_part * train_size)
#     valid_data = train_data[:valid_size]
#     train_data = train_data[valid_size:]
#     return train_data, test_data, valid_data

# sklearn
# def split_train_valid_test(data, train_size, valid_part=0.1):
#     length = len(data)
#     order = [i for i in range(length)]
#
#     train_data = []
#     test_data = []
#
#     data_y = []
#     for d in data:
#         data_y.append(d.y)
#
#     order_train, order_test, _, _ = train_test_split(order, data_y, test_size=(length - train_size), stratify=data_y)
#     for a in order_train:
#         train_data.append(data[a])
#     for b in order_test:
#         test_data.append(data[b])
#
#     random.shuffle(train_data)
#     valid_size = round(valid_part * train_size)
#     valid_data = train_data[:valid_size]
#     train_data = train_data[valid_size:]
#     return train_data, test_data, valid_data


def get_data_loader(dataset, save_path, batch_size, mini_batch_size):
    # param
    train_size = args[dataset]["train_size"]

    # load data
    inputs = np.load(f"{save_path}/{dataset}.inputs.npy")
    graphs = np.load(f"{save_path}/{dataset}.graphs.npy")

    # edge_type 中weight是type
    weights = np.load(f"{save_path}/{dataset}.weights.npy")
    targets = np.load(f"{save_path}/{dataset}.targets.npy")
    len_inputs = joblib.load(f"{save_path}/{dataset}.len.inputs.pkl")
    len_graphs = joblib.load(f"{save_path}/{dataset}.len.graphs.pkl")
    word2vec = np.load(f"{save_path}/{dataset}.word2vec.npy")

    poses = np.load(f"{save_path}/{dataset}.poses_pad.npy")
    pos2vec = np.load(f"{save_path}/{dataset}.pos2vec.npy")

    # py graph dtype
    data = []
    for x, edge_index, edge_type, y, lx, le, pos in tqdm(
            list(zip(inputs, graphs, weights, targets, len_inputs, len_graphs, poses))):
        # x就是输入的nodes
        x = torch.tensor(x[:lx], dtype=torch.long)
        # y就是标签
        y = torch.tensor(y, dtype=torch.long)

        pos = torch.tensor(pos[:lx], dtype=torch.long)
        edge_index = torch.tensor([e[:le] for e in edge_index], dtype=torch.long)
        edge_type = torch.tensor(edge_type[:le], dtype=torch.long)  # torch.float
        lens = torch.tensor(lx, dtype=torch.long)
        # x=nodes y=labels
        data.append(Data(x=x, y=y, edge_type=edge_type, edge_index=edge_index, length=lens, pos=pos))

    # split
    # train_data, test_data, valid_data = split_train_valid_test(data, train_size, valid_part=0.1)
    # v1
    train_data, test_data, valid_data = split_train_valid_test(data, dataset, train_size, valid_part=0.1)

    # return loader & word2vec
    return [MyDataLoader(data, batch_size=batch_size, mini_batch_size=mini_batch_size)
            for data in [train_data, test_data, valid_data]], word2vec, pos2vec
