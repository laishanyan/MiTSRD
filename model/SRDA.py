import torch
import torch.nn as nn
import torch.nn.functional as F

class SRDAttnModule(nn.Module):
    """
    Semantic Role Decomposition + Attention Module
    输入: 词嵌入 H: [batch_size, seq_len, embed_dim]
    输出: 融合后的特征 z: [batch_size, 2*hidden_dim]
    """
    def __init__(self, config):
        super(SRDAttnModule, self).__init__()
        # 通道投影矩阵
        self.proj_agent = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_experience = nn.Linear(config.hidden_size, config.hidden_size)
        # 注意力查询向量
        self.query_agent = nn.Parameter(torch.randn(config.hidden_size))
        self.query_experience = nn.Parameter(torch.randn(config.hidden_size))

    def forward(self, H):
        """
        H: [batch_size, seq_len, embed_dim]
        """
        # 通道投影
        H_A = self.proj_agent(H)         # [batch, seq_len, hidden_dim]
        H_E = self.proj_experience(H)    # [batch, seq_len, hidden_dim]

        # 注意力权重
        alpha = torch.softmax(torch.matmul(H_A, self.query_agent), dim=1)   # [batch, seq_len]
        beta = torch.softmax(torch.matmul(H_E, self.query_experience), dim=1) # [batch, seq_len]

        # 注意力加权求和
        z_A = torch.sum(H_A * alpha.unsqueeze(-1), dim=1)  # [batch, hidden_dim]
        z_E = torch.sum(H_E * beta.unsqueeze(-1), dim=1)   # [batch, hidden_dim]

        # 特征融合
        z = torch.cat([z_A, z_E], dim=-1)  # [batch, 2*hidden_dim]
        return z, z_A, z_E