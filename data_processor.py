import os
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
vi_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


def make_arg(span, words, label):
    if label == 'Y':
        label = 'predicate'
    if "." in label:
        label = label.lower()
    if not "Arg" in label:
        label = label.lower()
    if "Arg" in label:
        label = label.upper()
    span = [span[0], span[-1]]
    text = " ".join(words[span[0]: span[1]+1])
    return {'text': text, 'label': label, 'span': span}

def make_instance(tokens):
    # tách các cụm và từ có chứa nhãn ra theo bộ 4 -> viết dict:
    # text: t
    # words: [w1, w2 ,....]
    # label: l
    # span:(begin, end)

    words = [t[1] for t in tokens]
    max_arg = len(tokens[0][2])
    args = []
    for i in range(max_arg):
        arg_inline = [token[2][i] for token in tokens]
        span = []
        for i in range(len(arg_inline)):
            if arg_inline[i] != "_":
                if len(span) == 0:
                    span.append(i)
                if len(span) > 0:
                    if arg_inline[i] == arg_inline[span[-1]]:
                        span.append(i)
                    else:
                        label = arg_inline[span[0]]
                        args.append(make_arg(span, words, label))
                        span = []
                        span.append(i)
            else:
                if len(span) > 0:
                    label = arg_inline[span[0]]
                    args.append(make_arg(span, words, label))
                    span = []
    indices = set()
    for i in range(len(args)-1):
        for j in range(i+1, len(args)):
            if args[i]['span'] == args[j]['span']:
                if args[i]['label'] == args[j]['label']:
                    indices.add(j)
    for i in range(len(args)):
        if args[i]['label'] == 'predicate' or args[i]['label'].islower():
            indices.add(i)

    predicates = []
    for arg in args:
        if arg['label'].islower() and arg['label'] != 'predicate':
            predicates.append(arg)
    for _ in sorted(list(indices), reverse=True):
        args.pop(_)

    text = " ".join(words)

    BIO = make_BIO_label(words, predicates, args)
    instance = {'text': text, 'preds': predicates, 'args': args, 'tokens': words, 'BIO': BIO}
    return instance

def make_BIO_label(words, predicates, args):
    labels = []
    for pred in predicates:
        BIO = ["O" for _ in words]
        span = pred['span']
        for i in range(span[0], span[1]+1):
            BIO[i] = "P"
        labels.append(BIO)

    for arg in args:
        BIO = ["O" for _ in words]
        span = arg['span']
        BIO[span[0]] = "B-" + arg['label']
        for i in range(span[0]+1, span[1]+1):
            BIO[i] = "I-" + arg['label']
        labels.append(BIO)
    return labels

def make_vocab_label(train_instances, dev_instances, test_instances):
    label2id = {}
    id2label = {}

    for i in train_instances:
        labels = i['label']
        for l in labels:
            for i in l:
                if i not in label2id:
                    label2id[i] = 1
                else:
                    label2id[i] +=1

    for i in dev_instances:
        labels = i['label']
        for l in labels:
            for i in l:
                if i not in label2id:
                    label2id[i] = 1
                else:
                    label2id[i] +=1

    for i in test_instances:
        labels = i['label']
        for l in labels:
            for i in l:
                if i not in label2id:
                    label2id[i] = 1
                else:
                    label2id[i] +=1

    sorted_dict = dict(sorted(label2id.items(), key=lambda item: item[1], reverse=True))
    index = 0
    label2id = {}
    for k, v in sorted_dict.items():
        label2id[k] = index
        id2label[index] = k
        index += 1
    return label2id, id2label

def load_data(filename):
    data = []
    sent = []
    tokens = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == '' and len(tokens) != 0:
                sent.append(tokens)
                tokens = []
            else:
                dataline = line.split('\t')
                index = dataline[0]
                word = dataline[1]
                argm = dataline[12:]
                tokens.append([index, word, argm])
        if len(tokens) > 0:
            sent.append(tokens)
        data += sent
    return data

class Vi_SRL_processor:
    def __init__(self, pathdata, tokenizer):
        self.tokenizer = tokenizer
        self.instances = self.load_instances(pathdata)
    def load_dataset(self, pathdata):
        files = os.listdir(pathdata)
        train_file = ""
        test_file = ""
        dev_file = ""

        for file in files:
            if "train" in file:
                train_file = os.path.join(pathdata, file)
            if "dev" in file:
                dev_file = os.path.join(pathdata, file)
            if "test" in file:
                test_file = os.path.join(pathdata, file)

        train = load_data(train_file)
        dev = load_data(test_file)
        test = load_data(dev_file)
        return train, dev, test

    def load_instance(self, data):
        instances = []
        for sent in data:
            inst = make_instance(sent)
            text = inst['text']
            preds = inst['preds']
            args = inst['args']
            labels = inst['BIO']
            tokens = inst['tokens']
            indices = self.text_to_index(tokens)
            pred_indices = []
            arg_indices = []

            for pred in preds:
                label = pred['label']
                # ids_of_label = self.text_to_index(label)
                span = pred['span']
                indices_of_phase = [span[0]+1, span[1]+1]
                pred_indices.append([label, indices_of_phase])

            for arg in args:
                label = arg['label']
                # ids_of_label = self.text_to_index(label)
                span = arg['span']
                indices_of_phase = [span[0]+1, span[1]+1]
                arg_indices.append([label, indices_of_phase])
            instance = {"text": text, 'tokens': tokens,'label': labels, "indices": indices, "pred_indices": pred_indices, "arg_indices": arg_indices}
            instances.append(instance)
        return instances

    def make_dataframe(self, instances):
        sentences = []
        word_labels = []
        idx_sent = []
        idx_labels = []
        tokens = []
        for i in instances:
            sentences.append(i['text'])
            tokens.append(i['tokens'])
            labels = i['label']
            word_labels.append(labels)
            idx_sent.append(i['indices'])
            idx_label = []
            for label in labels:
                idx = [self.label2id[l] for l in label]
                idx_label.append(idx)
            idx_labels.append(idx_label)
        # print(instances)
        data = {'sentence': sentences,'tokens': tokens, 'word_labels': word_labels, "idx_sent": idx_sent, "idx_labels": idx_labels}
        df = pd.DataFrame(data)
        return df

    def load_instances(self, pathdata):
        train, dev, test = self.load_dataset(pathdata)
        train_instances = self.load_instance(train)
        dev_instances = self.load_instance(dev)
        test_instances = self.load_instance(test)
        self.label2id, self.id2label = make_vocab_label(train_instances, dev_instances, test_instances)
        train_df = self.make_dataframe(train_instances)
        dev_df = self.make_dataframe(dev_instances)
        test_df = self.make_dataframe(test_instances)

        return train_df, dev_df, test_df

    def text_to_index(self, tokenized_sentence):
        return self.tokenizer.convert_tokens_to_ids(tokenized_sentence)


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label2id):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id

    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        # sentence = self.data.sentence[index]
        word_labels = self.data.word_labels[index]
        tokenized_sentence = self.data.tokens[index]
        # idx_sent = self.data.idx_sent[index]
        # idx_labels = self.data.idx_labels[index]

        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        for i in range(len(word_labels)):
            word_labels[i].insert(0, "O") # add outside label for [CLS] token
            word_labels[i].insert(-1, "O") # add outside label for [SEP] token

        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
          # truncate
            tokenized_sentence = tokenized_sentence[:maxlen]
            for i in range(len(word_labels)):
                word_labels[i] = word_labels[i][:maxlen]

        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
          for i in range(len(word_labels)):
                word_labels[i] = word_labels[i] + ["O" for _ in range(maxlen - len(word_labels[i]))]

        # step 4: obtain the attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

        label_ids = []
        for label in word_labels:
            label_id = [self.label2id[l] for l in label]
            label_ids.append(label_id)

        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len


def make_like_NER(max_length_label, data, path):
    sentences = data["sentence"].tolist()
    word_labels = data["word_labels"].tolist()
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(sentences)):
            sent = sentences[i]
            m_label = word_labels[i]
            words = sent.split(" ")

            tokens = [[w] for w in words]
            if len(m_label) == 0:
                continue
                # for i in range(len(tokens)):
                #     tokens[i].append("O")

            for labels in m_label:
                for i in range(len(tokens)):
                    if labels[i] == "P":
                        labels[i] = "B-Predicate"
                    else:
                        if "-" in labels[i]:
                            split = labels[i].split('-')
                            a = "_".join(split[1:])
                            labels[i] = split[0] + "-" + a

                    tokens[i].append(labels[i])
            for token in tokens:
                while len(token) - 1 < max_length_label:
                    token.append("O")
                line = "\t".join(token) + "\n"
                f.write(line)
            f.write("\n")
    # return conll
def max_label(train, dev, test):
    m = 0
    word_labels = train["word_labels"].tolist()
    for labels in word_labels:
        if m < len(labels):
            m = len(labels)
    word_labels = dev["word_labels"].tolist()
    for labels in word_labels:
        if m < len(labels):
            m = len(labels)

    word_labels = test["word_labels"].tolist()
    for labels in word_labels:
        if m < len(labels):
            m = len(labels)
    return m
if __name__ == '__main__':
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 1
    LEARNING_RATE = 1e-05
    MAX_GRAD_NORM = 10

    pathdata = "dataset"
    viSRLprocessor = Vi_SRL_processor(pathdata, vi_tokenizer)

    tokenizer = viSRLprocessor.tokenizer
    label2id = viSRLprocessor.label2id
    id2label = viSRLprocessor.label2id
    train_df, dev_df, test_df = viSRLprocessor.instances

    max_length_label = max_label(train_df, dev_df, test_df)

    make_like_NER(max_length_label, train_df, '/home/dao/PycharmProjects/viSRL/layered-bilstm-crf/src/dataset/train.conll')
    make_like_NER(max_length_label, dev_df, '/home/dao/PycharmProjects/viSRL/layered-bilstm-crf/src/dataset/dev.conll')
    make_like_NER(max_length_label, test_df, '/home/dao/PycharmProjects/viSRL/layered-bilstm-crf/src/dataset/test.conll')

    # train_set = dataset(train_df, tokenizer, MAX_LEN, label2id)
    # dev_set = dataset(dev_df, tokenizer, MAX_LEN, label2id)
    # test_set = dataset(test_df, tokenizer, MAX_LEN, label2id)

    # print(train_set[0])
