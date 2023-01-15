import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import joblib
import numpy as np

dataset = "ohsumed"
save_path = 'temp_v1'
corpus_path = 'corpus_v1'

# param
stop_words = set(stopwords.words('english'))
# with open(f"stop_words.txt", "r", encoding="latin1") as f:
#     stop_words = set(f.read().strip().split("\n"))

# 词频
least_freq = 5
if dataset == "mr" or "SST" in dataset:
    stop_words = set()
    least_freq = 0


# func load texts & labels
def load_dataset(dataset):
    with open(f"{corpus_path}/{dataset}.texts.txt", "rb") as f:
        texts = f.read()
        texts = texts.decode('utf-8', 'ignore')
        texts = texts.strip().split("\n")
    with open(f"{corpus_path}/{dataset}.labels.txt", "r") as f:
        labels = f.read().strip().split("\n")
    return texts, labels


def filter_text_old(text: str):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)
    text = text.replace("'ll ", " will ")
    text = text.replace("'d ", " would ")
    text = text.replace("'m ", " am ")
    text = text.replace("'s ", " is ")
    text = text.replace("'re ", " are ")
    text = text.replace("'ve ", " have ")
    text = text.replace(" can't ", " can not ")
    text = text.replace(" ain't ", " are not ")
    text = text.replace("n't ", " not ")
    text = text.replace(",", " , ")
    text = text.replace("!", " ! ")
    text = text.replace("(", " ( ")
    text = text.replace(")", " ) ")
    text = text.replace("?", " ? ")
    text = re.sub(r"\s{2,}", " ", text)
    return " ".join(text.strip().split())

def filter_text(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# def filter_text(text: str):
#     text = text.lower()
#     text = text.replace("'ll ", " will ")
#     text = text.replace("'d ", " would ")
#     text = text.replace("'m ", " am ")
#     text = text.replace("'s ", " is ")
#     text = text.replace("'re ", " are ")
#     text = text.replace("'ve ", " have ")
#     text = text.replace(" can't ", " can not ")
#     text = text.replace(" ain't ", " are not ")
#     text = text.replace("n't ", " not ")
#     text = text.replace(". . .", " . ")
#     text = text.replace(" '", " ")
#     text = re.sub(r"\.{2,}", '.', text)  # 删除多余.
#     text = re.sub(r'\.$', '', text.strip())
#     text = re.sub(r'^\.', '', text.strip())
#     text = re.sub(r"[^A-Za-z0-9,.!?\'`]", " ", text)
#     text = text.replace(",", " , ")
#     text = text.replace("!", " ! ")
#     text = text.replace("?", " ? ")
#     text = text.replace("'", "")
#     text = re.sub(r"\s{2,}", " ", text)
#     return " ".join(text.strip().split())


def pos_text(text: str):
    pos = nltk.word_tokenize(text)
    return nltk.pos_tag(pos)


def words_pos_list(texts, text_pos, word2index):
    words_list = []
    pos_list = []
    for t, p in zip(texts, text_pos):
        temp = []
        temp_pos = []
        t_split = t.split()
        for i in range(0, len(t_split)):
            if t_split[i] in word2index:
                temp.append(t_split[i])
                temp_pos.append(p[i][1])
        words_list.append(temp)
        pos_list.append(temp_pos)
    return words_list, pos_list


if __name__ == '__main__':
    texts, labels = load_dataset(dataset)

    # handle texts
    texts_clean = [filter_text(t) for t in texts]
    text_pos = [pos_text(t) for t in texts_clean]

    word2count = Counter([w for t in texts_clean for w in t.split()])
    word_count = [[w, c] for w, c in word2count.items() if c >= least_freq and w not in stop_words]
    word2index = {w: i for i, (w, c) in enumerate(word_count)}

    # words_list = [[w for w in t.split() if w in word2index] for t in texts_clean]
    words_list, pos_list = words_pos_list(texts_clean, text_pos, word2index)

    pos2count = Counter([w for t in pos_list for w in t])
    pos_count = [[w, c] for w, c in pos2count.items()]
    pos2index = {w: i for i, (w, c) in enumerate(pos_count)}

    texts_remove = [" ".join(ws) for ws in words_list]

    # labels 2 targets
    label2index = {l: i for i, l in enumerate(set(labels))}
    targets = [label2index[l] for l in labels]

    # save
    with open(f"{save_path}/{dataset}.texts.clean.txt", "w") as f:
        f.write("\n".join(texts_clean))

    with open(f"{save_path}/{dataset}.texts.remove.txt", "w") as f:
        f.write("\n".join(texts_remove))

    np.save(f"{save_path}/{dataset}.targets.npy", targets)
    joblib.dump(word2index, f"{save_path}/{dataset}.word2index.pkl")
    joblib.dump(pos2index, f"{save_path}/{dataset}.pos2index.pkl")
    joblib.dump(pos_list, f"{save_path}/{dataset}.texts.pos.pkl")

    print('done')
