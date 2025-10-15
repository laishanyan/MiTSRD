import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGatedCrossAttention(nn.Module):
    def __init__(self, hidden_dim, config):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Cross-attention parameters
        self.W_q_char = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_sub = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_sub = nn.Linear(hidden_dim, hidden_dim)

        self.W_q_sub = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_char = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_char = nn.Linear(hidden_dim, hidden_dim)

        # Gate parameters
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, char_emb, subword_emb):
        """
        char_emb: (B, L_char, H)
        subword_emb: (B, L_sub, H)
        """
        B, L_char, H = char_emb.size()
        L_sub = subword_emb.size(1)

        # --- 1. Char attention on Subword ---
        Q_char = self.W_q_char(char_emb)          # (B, L_char, H)
        K_sub = self.W_k_sub(subword_emb)         # (B, L_sub, H)
        V_sub = self.W_v_sub(subword_emb)         # (B, L_sub, H)

        attn_scores = torch.matmul(Q_char, K_sub.transpose(1, 2)) / (H ** 0.5)  # (B, L_char, L_sub)
        attn_weights = F.softmax(attn_scores, dim=-1)
        char2sub = torch.matmul(attn_weights, V_sub)  # (B, L_char, H)

        # --- 2. Subword attention on Char ---
        Q_sub = self.W_q_sub(subword_emb)
        K_char = self.W_k_char(char_emb)
        V_char = self.W_v_char(char_emb)

        attn_scores2 = torch.matmul(Q_sub, K_char.transpose(1,2)) / (H ** 0.5)  # (B, L_sub, L_char)
        attn_weights2 = F.softmax(attn_scores2, dim=-1)
        sub2char = torch.matmul(attn_weights2, V_char)  # (B, L_sub, H)

        # --- 3. Gate fusion ---
        # 对齐 char_emb 和 sub2char (使用简单插值/映射到 char length)
        # 这里直接用 char_len 对齐
        if L_char != L_sub:
            sub2char = F.interpolate(sub2char.transpose(1,2), size=L_char, mode='linear', align_corners=False)
            sub2char = sub2char.transpose(1,2)  # (B, L_char, H)

        # Gate: sigmoid(W [char_emb; sub2char])
        gate_input = torch.cat([char2sub, sub2char], dim=-1)  # (B, L_char, 2H)
        gate = torch.sigmoid(self.gate_linear(gate_input))    # (B, L_char, H)

        fused = gate * char2sub + (1 - gate) * sub2char       # 动态融合
        return fused  # (B, L_char, H)