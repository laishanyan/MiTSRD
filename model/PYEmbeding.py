import torch
import torch.nn as nn
from pypinyin import pinyin, lazy_pinyin, Style
import numpy as np

class ShengmuYunmuPinyinEmbedding(nn.Module):
    def __init__(self, shengmu_vocab, yunmu_vocab, shengmu_dim=32, yunmu_dim=32):
        super().__init__()
        self.shengmu_vocab = shengmu_vocab
        self.yunmu_vocab = yunmu_vocab
        self.shengmu_embedding = nn.Embedding(len(shengmu_vocab), shengmu_dim, padding_idx=0)
        self.yunmu_embedding = nn.Embedding(len(yunmu_vocab), yunmu_dim, padding_idx=0)
        self.embedding_dim = shengmu_dim + yunmu_dim

    def forward(self, shengmu_indices, yunmu_indices):
        shengmu_emb = self.shengmu_embedding(shengmu_indices)
        yunmu_emb = self.yunmu_embedding(yunmu_indices)
        return torch.cat([shengmu_emb, yunmu_emb], dim=-1)  # (B, seq_len, H_pinyin)

def build_shengmu_yunmu_vocab():
    shengmu_list = ['<PAD>', 'b','p','m','f','d','t','n','l','g','k','h',
                    'j','q','x','zh','ch','sh','r','z','c','s','y','w']
    yunmu_list = ['<PAD>','a','o','e','i','u','v','ai','ei','ui','ao','ou',
                  'iu','ie','ve','er','an','en','in','un','vn','ang','eng','ing','ong']
    return {s:i for i,s in enumerate(shengmu_list)}, {y:i for i,y in enumerate(yunmu_list)}

def text_to_shengmu_yunmu_indices(text, pinyin_embedding):
    shengmu_list = lazy_pinyin(text, style=Style.INITIALS)
    yunmu_list = lazy_pinyin(text, style=Style.FINALS)
    shengmu_indices = torch.tensor([pinyin_embedding.shengmu_vocab.get(s, 0) for s in shengmu_list]).unsqueeze(0)
    yunmu_indices = torch.tensor([pinyin_embedding.yunmu_vocab.get(y, 0) for y in yunmu_list]).unsqueeze(0)
    sm = [0] + shengmu_indices.numpy().tolist()[0]
    ym = [0] + yunmu_indices.numpy().tolist()[0]
    return sm, ym

if __name__ == '__main__':
    shengmu_vocab, yunmu_vocab = build_shengmu_yunmu_vocab()
    shengmu_dim = 768
    yunmu_dim = 768
    text = '海南大学啊'
    pinyin_embedding = ShengmuYunmuPinyinEmbedding(shengmu_vocab, yunmu_vocab,
                                                   shengmu_dim, yunmu_dim)

    shengmu_indices, yunmu_indices = text_to_shengmu_yunmu_indices(text, pinyin_embedding)
    print(shengmu_indices, yunmu_indices)
    encode = pinyin_embedding(shengmu_indices, yunmu_indices)
    print(encode)


