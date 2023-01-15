import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import joblib
import numpy as np
from stanfordcorenlp import StanfordCoreNLP

dataset = "mr"

nlp = StanfordCoreNLP('.\stanford-corenlp-4.2.0')

# param
stop_words = set(stopwords.words('english'))
least_freq = 5
if dataset == "mr" or "SST" in dataset:
    stop_words = set()
    least_freq = 0


# func load texts & labels
def load_dataset(dataset):
    with open(f"corpus/{dataset}.texts.txt", "r", encoding="latin1") as f:
        texts = f.read().strip().split("\n")
    with open(f"corpus/{dataset}.labels.txt", "r") as f:
        labels = f.read().strip().split("\n")
    return texts, labels


def filter_text(text: str):
    text = text.lower()
    text = re.sub(r"([\w\.-]+)@([\w\.-]+)(\.[\w\.]+)", " ", text)  # 删除邮件地址
    text = re.sub(r"([\w\.-]+)@([\w\.-]+)", " ", text)  # 删除邮件地址
    text = re.sub(r"([\w\.-]+)(\.[\w\.]+)", " ", text)  # 删除网址
    text = text.replace("'ll ", " will ")
    text = text.replace("'d ", " would ")
    text = text.replace("'m ", " am ")
    text = text.replace("'s ", " is ")
    text = text.replace("'re ", " are ")
    text = text.replace("'ve ", " have ")
    text = text.replace(" can't ", " can not ")
    text = text.replace(" ain't ", " are not ")
    text = text.replace("n't ", " not ")
    text = text.replace(". . .", " . ")
    text = text.replace(" '", " ")
    text = re.sub(r"\.{2,}", '.', text)  # 删除多余.
    text = re.sub(r'\.$', '', text.strip())
    text = re.sub(r'^\.', '', text.strip())
    text = re.sub(r"[^A-Za-z0-9,.!?\'`]", " ", text)
    text = text.replace(",", " , ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace("'", "")
    # text = re.sub('[^a-z^A-Z]', ' ', text)  # 删除非英文字符
    text = re.sub(r"\s{2,}", " ", text)
    return " ".join(text.strip().split())


def pos_text(text: str):
    pos = nlp.word_tokenize(text)
    return nltk.pos_tag(pos)


def words_pos_list(texts, text_pos, word2index):
    words_list = []
    pos_list = []
    for t, p in zip(texts, text_pos):
        temp = []
        temp_pos = []
        t_split = nlp.word_tokenize(t)
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

    word2count = Counter([w for t in texts_clean for w in nlp.word_tokenize(t)])
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
    with open(f"temp/{dataset}.texts.clean.txt", "w") as f:
        f.write("\n".join(texts_clean))

    with open(f"temp/{dataset}.texts.remove.txt", "w") as f:
        f.write("\n".join(texts_remove))

    np.save(f"temp/{dataset}.targets.npy", targets)
    joblib.dump(word2index, f"temp/{dataset}.word2index.pkl")
    joblib.dump(pos2index, f"temp/{dataset}.pos2index.pkl")
    joblib.dump(pos_list, f"temp/{dataset}.texts.pos.pkl")

    print('done')
