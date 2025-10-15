# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
from .PYEmbeding import ShengmuYunmuPinyinEmbedding, build_shengmu_yunmu_vocab
from .DGAttention import DynamicGatedCrossAttention
from .SRDA import SRDAttnModule

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(2 * config.hidden_size, config.num_classes)
        self.shengmu_vocab, self.yunmu_vocab = build_shengmu_yunmu_vocab()
        self.pinyin_embedding = ShengmuYunmuPinyinEmbedding(self.shengmu_vocab, self.yunmu_vocab,
                                                       config.pinyin_dim, config.pinyin_dim)

        self.onedim_pool = nn.AvgPool1d(kernel_size=3)
        self.dgattention = DynamicGatedCrossAttention(hidden_dim=config.hidden_size, config=config)

        self.srda = SRDAttnModule(config)


    def word_embdding(self, word):
        batch = word.shape[0]
        word = word.view(batch,-1)
        word_emb, text_cls = self.bert(word, output_all_encoded_layers=False)
        out = self.onedim_pool(word_emb.transpose(1, 2))
        return out.transpose(1, 2)

    def chart_embedding(self, chart, mask):
        encoder_chart, text_cls = self.bert(chart, attention_mask=mask, output_all_encoded_layers=False)
        return encoder_chart


    def forward(self, x):
        chart = x[0]
        py_encode = self.pinyin_embedding(x[1], x[2])
        word = x[3]
        mask = x[5]  # 对adding部分进行mpask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_chart = self.chart_embedding(chart, mask)
        encoder_word = self.word_embdding(word)
        out_chart = encoder_chart + py_encode
        out = self.dgattention(out_chart, encoder_word)
        out = self.dropout(out)
        out, z_A, z_E = self.srda(out)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out, z_A, z_E
