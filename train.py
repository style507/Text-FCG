import time
from sklearn import metrics
from torch import nn
import torch
from config import *
from dataset import get_data_loader
from model import Model
from torch.optim import lr_scheduler

from focal_loss import FocalLoss, alpha_avg, alpha_max
import numpy as np
from collections import Counter


def train_eval(cate, loader, model, optimizer, loss_func, device):
    # 训练还是验证
    model.train() if cate == "train" else model.eval()
    # 预测 标签 loss
    preds, labels, loss_sum = [], [], 0.
    # 导入数据集
    for i in range(len(loader)):
        loss = torch.tensor(0., requires_grad=True).float().to(device)
        # loader[i]
        for j, graph in enumerate(loader[i]):
            # 将图输入gpu
            graph = graph.to(device)
            # tensor([1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1,
            #         1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,
            #         1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], device='cuda:0')
            targets = graph.y
            # Batch(batch=[1250], edge_attr=[7331], edge_index=[2, 7331], length=[64], x=[1250], y=[64])
            # 预测标签
            y = model(graph)
            # 计算loss
            loss = loss + loss_func(y, targets)
            # 列表添加预测和实际标签
            preds.append(y.max(dim=1)[1].data)
            labels.append(targets.data)

        loss = loss / len(loader[i])
        # 反向传播
        if cate == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

        loss_sum += loss.data

    preds = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    loss = loss_sum / len(loader)
    acc = metrics.accuracy_score(labels, preds) * 100
    return loss, acc, preds, labels


if __name__ == '__main__':
    dataset = "mr"
    save_path = 'nlp_temp_type'

    print("load dataset")
    # params
    batch_size = 128  # 反向传播时的batch
    mini_batch_size = 64  # 计算时的batch
    lr = 0.0003
    dropout = 0.5
    weight_decay = 0.000005
    in_dim = 300
    hid_dim = 128
    freeze = True
    step = 2
    start = 0

    num_classes = args[dataset]['num_classes']
    # [data, batch_size, mini_batch_size]
    (train_loader, test_loader, valid_loader), word2vec, pos2vec = get_data_loader(dataset, save_path,
                                                                                   batch_size, mini_batch_size)
    num_words = len(word2vec) - 1

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = Model(num_words, num_classes, word2vec=word2vec, in_dim=in_dim, hid_dim=hid_dim, step=step, freeze=freeze)

    # targets = np.load(f"temp/{dataset}.targets.npy")
    # alpha = alpha_avg(dict(Counter(targets)))
    # alpha = alpha_max(dict(Counter(targets)))
    # loss_func = FocalLoss(alpha=alpha, gamma=2, reduction='mean')

    # class_weights = torch.FloatTensor(alpha).to(device)
    # loss_func = nn.CrossEntropyLoss(weight=class_weights)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, 10, 0.9)

    model = model.to(device)

    print("-" * 50)
    print(model)
    print("-" * 50)
    print(dataset)
    print("-" * 50)
    print(f"params: [start={start}, dataset_path={save_path}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}]")
    print("-" * 50)

    best_acc = 0.
    best_test_acc = 0.
    preds = []
    labels = []
    for epoch in range(start + 1, 200):
        t1 = time.time()
        train_loss, train_acc, _, _ = train_eval("train", train_loader, model, optimizer, loss_func, device)
        valid_loss, valid_acc, _, _ = train_eval("valid", valid_loader, model, optimizer, loss_func, device)
        test_loss, test_acc, pred, label = train_eval("test", test_loader, model, optimizer, loss_func, device)

        if best_acc < valid_acc:
            best_acc = valid_acc
            best_test_acc = test_acc
            preds = pred
            labels = label

        cost = time.time() - t1
        print((f"epoch={epoch:03d}, cost={cost:.2f}, "
               f"train:[{train_loss:.4f}, {train_acc:.2f}%], "
               f"valid:[{valid_loss:.4f}, {valid_acc:.2f}%], "
               f"test:[{test_loss:.4f}, {test_acc:.2f}%], "
               f"best_valid={best_acc:.2f}%, test={best_test_acc:.2f}%"))

    print("Test Precision, Recall and F1-Score...")
    print(metrics.classification_report(labels, preds, digits=4))
    print("Macro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
    print("Micro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))
