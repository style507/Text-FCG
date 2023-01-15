import joblib
from tqdm import tqdm
import scipy.sparse as sp
from collections import Counter
import numpy as np
import itertools

# 数据集
dataset = "mr"

# 参数
window_size = 3
embedding_dim = 300
max_text_len = 300


# normalize
def normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    # 忽略除零
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj_normalized


def pad_seq(seq, pad_len):
    if len(seq) > pad_len:
        return seq[:pad_len]
    return seq + [0] * (pad_len - len(seq))


def check_same(target, words):
    b = []
    for index, nums in enumerate(words):
        if nums == target:
            b.append(index)
    return b


def connect_same(words):
    temp_edges = []
    list_set_words = list(set(words))
    for l in list_set_words:
        temp = check_same(l, words)
        if len(temp) > 1:
            temp_edges += list(itertools.permutations(temp, 2))
    return temp_edges


if __name__ == '__main__':
    # 词典和对应序号
    word2index = joblib.load(f"temp/{dataset}.word2index.pkl")
    # 数据集
    with open(f"temp/{dataset}.texts.remove.txt", "r") as f:
        texts = f.read().strip().split("\n")

    # 建图
    inputs = []
    graphs = []

    for text in tqdm(texts):
        words = [word2index[w] for w in text.split()]
        words = words[:max_text_len]

        nodes = words
        edges = []
        for i in range(len(words)):
            center = i
            for j in range(i - window_size, i + window_size + 1):
                if i != j and 0 <= j < len(words):
                    neighbor = j
                    edges.append((center, neighbor))
        cs = connect_same(words)
        edges += cs
        edge_count = Counter(edges).items()
        # 邻接矩阵
        row = [x for (x, y), c in edge_count]
        col = [y for (x, y), c in edge_count]
        weight = [c for (x, y), c in edge_count]
        # 归一化
        adj = sp.csr_matrix((weight, (row, col)), shape=(len(nodes), len(nodes)))
        adj_normalized = normalize_adj(adj)
        weight_normalized = [adj_normalized[x][y] for (x, y), c in edge_count]
        # 保存节点和图
        inputs.append(nodes)
        graphs.append([row, col, weight_normalized])

    len_inputs = [len(e) for e in inputs]
    len_graphs = [len(x) for x, y, c in graphs]

    # padding input
    pad_len_inputs = max(len_inputs)
    pad_len_graphs = max(len_graphs)
    inputs_pad = [pad_seq(e, pad_len_inputs) for e in tqdm(inputs)]
    graphs_pad = [[pad_seq(ee, pad_len_graphs) for ee in e] for e in tqdm(graphs)]

    inputs_pad = np.array(inputs_pad)
    weights_pad = np.array([c for x, y, c in graphs_pad])
    graphs_pad = np.array([[x, y] for x, y, c in graphs_pad])

    # word2vec
    all_vectors = np.load(f"source/glove.6B.{embedding_dim}d.npy")
    all_words = joblib.load(f"source/glove.6B.words.pkl")
    all_word2index = {w: i for i, w in enumerate(all_words)}
    index2word = {i: w for w, i in word2index.items()}
    word_set = [index2word[i] for i in range(len(index2word))]
    oov = np.random.normal(-0.001, 0.001, embedding_dim)
    word2vec = [all_vectors[all_word2index[w]] if w in all_word2index else oov for w in word_set]
    word2vec.append(np.zeros(embedding_dim))

    # save
    joblib.dump(len_inputs, f"temp/{dataset}.len.inputs.pkl")
    joblib.dump(len_graphs, f"temp/{dataset}.len.graphs.pkl")
    np.save(f"temp/{dataset}.inputs.npy", inputs_pad)
    np.save(f"temp/{dataset}.graphs.npy", graphs_pad)
    np.save(f"temp/{dataset}.weights.npy", weights_pad)
    np.save(f"temp/{dataset}.word2vec.npy", word2vec)

    print('done')