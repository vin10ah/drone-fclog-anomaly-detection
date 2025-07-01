import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CompilerFriendlyAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(attn_output)

class CompilerFriendlyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        self.attn = CompilerFriendlyAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = F.gelu(self.linear1(x))
        ffn_out = self.dropout1(ffn_out)
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        ffn_out = x + ffn_out
        # ffn_out = self.norm2(ffn_out)

        return ffn_out

class FTTransformerTimeSeries(nn.Module):
    def __init__(self, num_numerical_features, d_token=64, n_blocks=4, n_heads=4, dropout=0.2, n_classes=1, max_len=500):
        super().__init__()
        self.d_token = d_token
        self.input_proj = nn.Linear(num_numerical_features * 2, d_token)  # original + diff
        self.input_norm = nn.LayerNorm(d_token)
        self.input_dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_token, max_len=max_len)

        self.blocks = nn.ModuleList([
            CompilerFriendlyTransformerBlock(d_token, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        self.output_norm = nn.LayerNorm(d_token)
        self.output_linear1 = nn.Linear(d_token, d_token)
        self.output_linear2 = nn.Linear(d_token, n_classes)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, T, F)
        x_diff = x[:, 1:, :] - x[:, :-1, :]
        zero = torch.zeros(x.size(0), 1, x.size(2), device=x.device)  # (B, 1, F)
        x_diff = torch.cat([zero, x_diff], dim=1) 
        mean = x_diff.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        var = ((x_diff - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        alpah = 1e-6
        std = torch.sqrt(var + alpah)
        x_diff = x_diff / std
        x = torch.cat([x, x_diff], dim=-1)  # (B, T, F*2)

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        x = self.input_dropout(x)
        x = self.pos_encoder(x)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=1)  # global average pooling
        x = self.output_norm(x)
        x = F.gelu(self.output_linear1(x))
        x = self.output_dropout(x)
        x = self.output_linear2(x)
        return x