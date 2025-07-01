import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, ff_mult=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # First feed-forward block with residual
        residual = x
        x = self.norm1(x)
        x = self.ffn1(x)
        x = self.dropout1(x)
        x = residual + x
        
        # Second feed-forward block with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn2(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x

class FTTransformer(nn.Module):
    def __init__(self, num_numerical_features=8):
        super().__init__()
        self.input_layer = nn.Linear(num_numerical_features, 128)
        self.encoder = nn.Sequential(
            SimpleTransformerBlock(dim=128, ff_mult=2, dropout=0.1),
            SimpleTransformerBlock(dim=128, ff_mult=2, dropout=0.1),
            SimpleTransformerBlock(dim=128, ff_mult=2, dropout=0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
        )
        
    def forward(self, x_num):
        x = self.input_layer(x_num)  # (B, 128)
        x = self.encoder(x)  # Feed-forward blocks
        x = self.head(x)  # Final output
        return x
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, ff_mult=2, dropout=0.1):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
#         self.norm1 = nn.LayerNorm(dim)
#         self.dropout1 = nn.Dropout(dropout)

#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * ff_mult),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim * ff_mult, dim),
#         )
#         self.norm2 = nn.LayerNorm(dim)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x):
#         # Self-attention with residual
#         attn_out, _ = self.attn(x, x, x, need_weights=False)
#         attn_out = self.dropout1(attn_out)
#         x = x + attn_out  # Inline residual
#         x = self.norm1(x)

#         # Feed-forward with residual
#         ffn_out = self.ffn(x)
#         ffn_out = self.dropout2(ffn_out)
#         x = x + ffn_out  # Inline residual
#         x = self.norm2(x)

#         return x

# class FTTransformer(nn.Module):
#     def __init__(self, num_numerical_features=8):
#         super().__init__()
#         self.input_layer = nn.Linear(num_numerical_features, 128)

#         self.encoder = nn.Sequential(
#             TransformerBlock(dim=128, num_heads=4, ff_mult=2, dropout=0.1),
#             TransformerBlock(dim=128, num_heads=4, ff_mult=2, dropout=0.1),
#         )

#         self.head = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.LeakyReLU(),
#             nn.Linear(64, 1),
#         )

#     def forward(self, x_num):
#         x = self.input_layer(x_num)  # (B, 128)
#         x = x.unsqueeze(1)           # (B, 1, 128) → seq_len=1로 설정
#         x = self.encoder(x)          # 통과 가능
#         x = x.squeeze(1)             # (B, 128)로 다시 복원
#         x = self.head(x)             # 최종 출력
#         return x



# class FTTransformer(nn.Module):
#     def __init__(self, num_numerical_features, d_token=64, n_blocks=4, n_heads=8, dropout=0.1, n_classes=1):
#         super().__init__()

#         self.d_token = d_token
#         self.num_numerical_features = num_numerical_features

#         # 수치형 특징을 각각 token화
#         self.num_linear = nn.Linear(1, d_token)

#         # Custom Transformer Encoder Blocks
#         self.blocks = nn.ModuleList([
#             TransformerBlock(d_token, n_heads, d_token * 4, dropout)
#             for _ in range(n_blocks)
#         ])

#         # MLP Head
#         self.head = nn.Sequential(
#             nn.LayerNorm(d_token),
#             nn.Linear(d_token, d_token),
#             nn.LeakyReLU(),
#             nn.Linear(d_token, n_classes)
#         )

#     def forward(self, x_num):

#         # (B, F, 1) → (B, F, d_token)
#         x_num = x_num.unsqueeze(-1)
#         x_tokens = self.num_linear(x_num)  # (B, F, d_token)

#         x = x_tokens

#         # Transformer blocks
#         for block in self.blocks:
#             x = block(x)  # (B, F, d_token)

#         # 평균 풀링으로 전체 시퀀스 요약
#         pooled = x.mean(dim=1)  # (B, d_token)

#         # MLP Head
#         out = self.head(pooled)  # (B, n_classes)

#         return out
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ONNXFriendlyMultiHeadAttention(nn.Module):
#     def __init__(self, d_model, n_heads, dropout=0.1):
#         super().__init__()
#         assert d_model % n_heads == 0
#         self.n_heads = n_heads
#         self.d_head = d_model // n_heads

#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)

#         self.out_proj = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         B, T, D = x.shape  # (batch, seq_len, d_model)
#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)

#         # reshape for multihead
#         q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, T, d)
#         k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
#         v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
#         attn_probs = torch.softmax(attn_scores, dim=-1)
#         attn_probs = self.dropout(attn_probs)

#         context = torch.matmul(attn_probs, v)  # (B, h, T, d)
#         context = context.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, D)

#         return self.out_proj(context)


# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
#         super().__init__()
#         self.attn = ONNXFriendlyMultiHeadAttention(d_model, nhead, dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)

#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, dim_feedforward),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim_feedforward, d_model)
#         )
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.norm1(x + self.dropout1(self.attn(x)))
#         x = self.norm2(x + self.dropout2(self.ffn(x)))
#         return x


# class FTTransformer(nn.Module):
#     def __init__(self, num_numerical_features, d_token=64, n_blocks=4, n_heads=8, dropout=0.1, n_classes=1):
#         super().__init__()

#         self.d_token = d_token
#         self.num_numerical_features = num_numerical_features

#         # 수치형 특징을 각각 token화
#         self.num_linear = nn.Linear(1, d_token)

#         # CLS token
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

#         # Custom Transformer Encoder Blocks
#         self.blocks = nn.ModuleList([
#             TransformerBlock(d_token, n_heads, d_token * 4, dropout)
#             for _ in range(n_blocks)
#         ])

#         # MLP Head
#         self.head = nn.Sequential(
#             nn.LayerNorm(d_token),
#             nn.Linear(d_token, d_token),
#             nn.LeakyReLU(),
#             nn.Linear(d_token, n_classes)
#         )

#     def forward(self, x_num):

#         # (B, F, 1) → (B, F, d_token)
#         x_num = x_num.unsqueeze(-1)
#         x_tokens = self.num_linear(x_num)  # (B, F, d_token)

#         x = x_tokens

#         # Transformer blocks
#         for block in self.blocks:
#             x = block(x)  # (B, F, d_token)

#         # 평균 풀링으로 전체 시퀀스 요약
#         pooled = x.mean(dim=1)  # (B, d_token)

#         # MLP Head
#         out = self.head(pooled)  # (B, n_classes)

#         return out
# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout)

#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, dim_feedforward),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim_feedforward, d_model)
#         )
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x):
#         # Self-attention
#         attn_output, _ = self.attn(x, x, x, attn_mask=None, key_padding_mask=None, need_weights=False)
#         x = self.norm1(x + self.dropout1(attn_output))

#         # Feedforward
#         ffn_output = self.ffn(x)
#         x = self.norm2(x + self.dropout2(ffn_output))

#         return x


# class FTTransformer(nn.Module):
#     def __init__(self, num_numerical_features, d_token=64, n_blocks=4, n_heads=8, dropout=0.1, n_classes=1):
#         super().__init__()

#         self.d_token = d_token
#         self.num_numerical_features = num_numerical_features

#         # 수치형 특징을 각각 token화
#         self.num_linear = nn.Linear(1, d_token)

#         # CLS token
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

#         # Custom Transformer Encoder Blocks
#         self.blocks = nn.ModuleList([
#             TransformerBlock(d_token, n_heads, d_token * 4, dropout)
#             for _ in range(n_blocks)
#         ])

#         # MLP Head
#         self.head = nn.Sequential(
#             nn.LayerNorm(d_token),
#             nn.Linear(d_token, d_token),
#             nn.LeakyReLU(),
#             nn.Linear(d_token, n_classes)
#         )

#     def forward(self, x_num):
#         B, F = x_num.shape

#         # (B, F, 1) → (B, F, d_token)
#         x_num = x_num.unsqueeze(-1)
#         x_tokens = self.num_linear(x_num)  # (B, F, d_token)

#         x = x_tokens

#         # Transformer blocks
#         for block in self.blocks:
#             x = block(x)  # (B, F, d_token)

#         # 평균 풀링으로 전체 시퀀스 요약
#         pooled = x.mean(dim=1)  # (B, d_token)

#         # MLP Head
#         out = self.head(pooled)  # (B, n_classes)

#         return out