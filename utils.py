# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
from pypinyin import lazy_pinyin, Style
from model.PYEmbeding import build_shengmu_yunmu_vocab, ShengmuYunmuPinyinEmbedding, text_to_shengmu_yunmu_indices

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

def get_pinyin_ids(config, text):
    shengmu_vocab, yunmu_vocab = build_shengmu_yunmu_vocab()
    pinyin_embedding = ShengmuYunmuPinyinEmbedding(shengmu_vocab, yunmu_vocab,
                                                   config.pinyin_dim, config.pinyin_dim)

    shengmu_indices, yunmu_indices = text_to_shengmu_yunmu_indices(text, pinyin_embedding)
    return shengmu_indices, yunmu_indices

def remove_list(list_data):
    # 删除无效子词
    index_list = []
    for i in range(len(list_data)):
        if len(list_data[i]) != 3:
            index_list.append(i)
    for ind in index_list:
        del list_data[ind]
    return list_data

def build_dataset(config):

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                content_list = list(content)
                shengmu_ids, yunmu_ids = get_pinyin_ids(config, content)
                token = config.tokenizer.tokenize(content)
                word_token = config.tokenizer(content_list)['input_ids']
                word_token = remove_list(word_token)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                        shengmu_ids += ([0] * (pad_size - len(shengmu_ids)))
                        yunmu_ids += ([0] * (pad_size - len(yunmu_ids)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        shengmu_ids = shengmu_ids[:pad_size]
                        yunmu_ids = yunmu_ids[:pad_size]
                        word_token = word_token[:pad_size]
                        seq_len = pad_size
                if pad_size:
                    if len(word_token) < pad_size:
                        word_token += ([[101, 0, 0]] * (pad_size - len(word_token)))
                    else:
                        word_token = word_token[:pad_size]

                contents.append((token_ids, shengmu_ids, yunmu_ids, word_token, int(label), seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        sm = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        ym = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        word = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[4] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        return (x, sm, ym, word, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
