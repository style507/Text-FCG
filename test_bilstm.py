import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn import metrics

import joblib
import numpy as np
from tqdm import tqdm
from config import args
import time

dataset = "mr"

word2vec = np.load(f"temp/{dataset}.word2vec.npy")
targets = np.load(f"temp/{dataset}.targets.npy")
word2index = joblib.load(f"temp/{dataset}.word2index.pkl")
with open(f"temp/{dataset}.texts.remove.txt", "r") as f:
    texts = f.read().strip().split("\n")


class BiLstm(nn.Module):
    def __init__(self, num_words, out_dim, in_dim, hid_dim, dropout=0.5, bias=False, bidirectional=True, word2vec=None,
                 freeze=True):
        super(BiLstm, self).__init__()

        self.bias = bias
        self.hid_dim = hid_dim

        if word2vec is None:
            self.embed = nn.Embedding(num_words + 1, in_dim, num_words)
        else:
            self.embed = torch.nn.Embedding.from_pretrained(torch.from_numpy(word2vec).float(), num_words)

        self.lstm = nn.LSTM(in_dim, hid_dim, batch_first=True, bidirectional=bidirectional)

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, out_dim, bias=True)
        )
        self.dp = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hid_dim * 2, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        out = self.dp(out)
        out = F.relu(self.fc1(out))
        out = F.avg_pool2d(out, (out.shape[1], 1)).squeeze()
        out = self.fc2(out)
        return out


def dataloader(text_list, targets, batch_size):
    x_s = []
    for t in tqdm(text_list):
        if len(t) <= 30:
            t = t + [0] * (30 - len(t))  # max_len=30
        x_s.append(t)

    # x就是输入的nodes
    x = torch.tensor(x_s, dtype=torch.long)
    # y就是标签
    y = torch.tensor(targets, dtype=torch.long)

    data = TensorDataset(x, y)
    # split
    train_data, test_data = random_split(dataset=data, lengths=[args[dataset]["train_size"],
                                                                len(text_list)-args[dataset]["train_size"]])
    # return loader
    return [DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
            for data in [train_data, test_data]]


def train_eval(cate, loader, model, optimizer, loss_func, device):
    # 训练还是验证
    model.train() if cate == "train" else model.eval()
    # 预测 标签 loss
    preds, labels, loss_sum = [], [], 0.
    # 导入数据集
    for i in range(len(loader)):
        loss = torch.tensor(0., requires_grad=True).float().to(device)
        # loader[i]
        for j, (data, target) in enumerate(loader):
            targets = target
            # 预测标签
            y = model(data)
            # 计算loss
            loss = loss + loss_func(y, targets)
            # 列表添加预测和实际标签
            preds.append(y.max(dim=1)[1].data)
            labels.append(targets.data)

        loss = loss / len(loader)
        # 反向传播
        if cate == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_sum += loss.data

    preds = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    loss = loss_sum / len(loader)
    acc = metrics.accuracy_score(labels, preds) * 100
    return loss, acc, preds, labels


if __name__ == '__main__':
    inputs = []
    for text in tqdm(texts):
        # 文本转化成序号
        words = [word2index[w] for w in text.split()]
        # 限制最大长度
        words = words[:30]
        inputs.append(words)

    num_words = len(word2vec) - 1
    num_classes = args[dataset]['num_classes']

    batch_size = 4096  # 反向传播时的batch
    mini_batch_size = 64  # 计算时的batch
    lr = 0.01
    dropout = 0.5
    weight_decay = 0.
    hid_dim = 96
    freeze = True
    start = 0

    (train_loader, test_loader) = dataloader(inputs, targets, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLstm(num_words, num_classes, in_dim=100, hid_dim=hid_dim, dropout=0.5, bias=False, bidirectional=True,
                   word2vec=word2vec)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)

    best_acc = 0.
    for epoch in range(start + 1, 100):
        t1 = time.time()
        train_loss, train_acc, _, _ = train_eval("train", train_loader, model, optimizer, loss_func, device)
        test_loss, test_acc, preds, labels = train_eval("test", test_loader, model, optimizer, loss_func, device)

        if best_acc < test_acc:
            best_acc = test_acc

        cost = time.time() - t1
        print((f"epoch={epoch:03d}, cost={cost:.2f}, "
               f"train:[{train_loss:.4f}, {train_acc:.2f}%], "
               f"test:[{test_loss:.4f}, {test_acc:.2f}%], "
               f"best_acc={best_acc:.2f}%"))

    print("Test Precision, Recall and F1-Score...")
    print(metrics.classification_report(labels, preds, digits=4))
    print("Macro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
    print("Micro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))
