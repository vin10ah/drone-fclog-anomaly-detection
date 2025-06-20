import torch
import torch.nn as nn
import math


class CompilerFriendlyAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        
        # 별도의 Q, K, V projection (더 안정적)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, D = x.size()
        
        # 개별적으로 Q, K, V 계산 (chunk와 squeeze 피하기)
        q = self.q_proj(x)  # (B, T, D)
        k = self.k_proj(x)  # (B, T, D)
        v = self.v_proj(x)  # (B, T, D)
        
        # 고정된 view 연산 사용 (동적 계산 피하기)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        
        # Attention 계산
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)  # (B, H, T, d_head)
        
        # 다시 원래 형태로 (고정된 연산)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, H, d_head)
        attn_output = attn_output.view(B, T, D)  # (B, T, D)
        
        return self.out_proj(attn_output)

class CompilerFriendlyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        
        self.attn = CompilerFriendlyAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN을 Sequential 대신 개별 레이어로
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Post-norm (더 안정적)
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.linear1(x)
        ffn_out = torch.relu(ffn_out)
        ffn_out = self.dropout1(ffn_out)
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        
        x = self.norm2(x + ffn_out)
        return x



class CompilerFriendlySAINTBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        # Col-Attn
        self.col_attn = CompilerFriendlyAttention(d_model, n_heads, dropout)
        self.col_norm = nn.LayerNorm(d_model)
        
        # Row-Attn
        self.row_attn = CompilerFriendlyAttention(d_model, n_heads, dropout)
        self.row_norm = nn.LayerNorm(d_model)
        
        # FFN layers (2번)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_ffn1 = nn.LayerNorm(d_model)
        
        self.linear3 = nn.Linear(d_model, d_model * 4)
        self.linear4 = nn.Linear(d_model * 4, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # ---- Col Attention ----
        col_out = self.col_attn(x)
        x = self.col_norm(x + col_out)
        
        # ---- FFN 1 ----
        ffn1 = self.linear1(x)
        ffn1 = torch.relu(ffn1)
        ffn1 = self.dropout1(ffn1)
        ffn1 = self.linear2(ffn1)
        ffn1 = self.dropout2(ffn1)
        x = self.norm_ffn1(x + ffn1)
        
        # ---- Row Attention ----
        x_t = x.transpose(0, 1)  # (B, T, D) -> (T, B, D)
        row_out = self.row_attn(x_t)
        x_t = self.row_norm(x_t + row_out)
        x = x_t.transpose(0, 1)  # (T, B, D) -> (B, T, D)
        
        # ---- FFN 2 ----
        ffn2 = self.linear3(x)
        ffn2 = torch.relu(ffn2)
        ffn2 = self.dropout3(ffn2)
        ffn2 = self.linear4(ffn2)
        ffn2 = self.dropout4(ffn2)
        x = self.norm_ffn2(x + ffn2)
        
        return x


class CompilerFriendlySAINT(nn.Module):
    def __init__(self, num_categories=None, num_numericals=0, d_model=32, n_heads=4,
                 num_classes=1, num_layers=2, dropout=0.2):
        super().__init__()
        self.has_cat = bool(num_categories)
        self.has_num = num_numericals > 0
        self.d_model = d_model
        
        # Categorical embeddings
        if self.has_cat:
            self.cat_embeds = nn.ModuleList([
                nn.Embedding(cat_size, d_model) for cat_size in num_categories
            ])
        
        # Numerical linear projection
        if self.has_num:
            self.num_linear = nn.Linear(num_numericals, d_model)
        
        # SAINT Blocks
        self.blocks = nn.ModuleList([
            CompilerFriendlySAINTBlock(d_model, n_heads, dropout) for _ in range(num_layers)
        ])
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x_cat=None, x_num=None):
        tokens = []
        
        if self.has_cat and x_cat is not None:
            cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)]
            cat_embeds = torch.stack(cat_embeds, dim=1)  # (B, F_cat, D)
            tokens.append(cat_embeds)
        
        if self.has_num and x_num is not None:
            num_embed = self.num_linear(x_num).unsqueeze(1)  # (B, 1, D)
            tokens.append(num_embed)
        
        x = torch.cat(tokens, dim=1)  # (B, F, D)
        
        for block in self.blocks:
            x = block(x)
        
        x = x.mean(dim=1)  # Global Average Pooling
        return self.head(x)


# class FeedForward(nn.Module):
#     def __init__(self, emb_dim, dim_feedforward, dropout=0.1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(emb_dim, dim_feedforward),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(dim_feedforward, emb_dim),
#             nn.Dropout(dropout)
#         )
#         self.norm = nn.LayerNorm(emb_dim)

#     def forward(self, x):
#         return self.norm(x + self.net(x))
    

# class SAINTBlock(nn.Module):
#     def __init__(self, emb_dim, num_heads, dim_feedforward, dropout=0.1):
#         super().__init__()
#         self.col_attn = CustomMultiHeadSelfAttention(emb_dim, num_heads, dropout)
#         self.ffn1 = FeedForward(emb_dim, dim_feedforward, dropout)
#         self.row_attn = CustomMultiHeadSelfAttention(emb_dim, num_heads, dropout)
#         self.ffn2 = FeedForward(emb_dim, dim_feedforward, dropout)

#     def forward(self, x):
#         # Column-wise attention
#         x = self.col_attn(x)
#         x = self.ffn1(x)

#         # Row-wise attention: swap batch & feature dims
#         x = x.transpose(0, 1)  # (F, B, D)
#         x = self.row_attn(x)
#         x = x.transpose(0, 1)  # (B, F, D)

#         x = self.ffn2(x)
#         return x

# class SAINT(nn.Module):
#     def __init__(self, num_categories=None, num_numericals=0, emb_dim=32, num_heads=4,
#                  num_classes=1, num_layers=2, dim_feedforward=256, dropout=0.1):
#         super().__init__()
#         self.has_cat = bool(num_categories)
#         self.has_num = num_numericals > 0
#         self.emb_dim = emb_dim

#         # Categorical feature embeddings
#         if self.has_cat:
#             self.cat_embeddings = nn.ModuleList([
#                 nn.Embedding(num_cat, emb_dim) for num_cat in num_categories
#             ])

#         # Numerical feature projection
#         if self.has_num:
#             self.num_linear = nn.Linear(num_numericals, emb_dim)

#         # SAINT Blocks (Col+Row Attention)
#         self.blocks = nn.ModuleList([
#             SAINTBlock(emb_dim, num_heads, dim_feedforward, dropout)
#             for _ in range(num_layers)
#         ])

#         # Classification head
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(emb_dim),
#             nn.Linear(emb_dim, emb_dim),
#             nn.ReLU(),
#             nn.Linear(emb_dim, num_classes)
#         )

#     def forward(self, x_cat=None, x_num=None):
#         tokens = []

#         if self.has_cat and x_cat is not None:
#             cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
#             cat_embeds = torch.stack(cat_embeds, dim=1)  # (B, N_cat, D)
#             tokens.append(cat_embeds)

#         if self.has_num and x_num is not None:
#             num_embed = self.num_linear(x_num).unsqueeze(1)  # (B, 1, D)
#             tokens.append(num_embed)

#         # Combine
#         x = torch.cat(tokens, dim=1)  # (B, F, D)

#         for block in self.blocks:
#             x = block(x)

#         # Global average pooling
#         x = x.mean(dim=1)  # (B, D)

#         return self.classifier(x)  # (B, num_classes)





# class CustomMultiHeadSelfAttention(nn.Module):
#     def __init__(self, emb_dim, num_heads, dropout=0.1):
#         super().__init__()
#         assert emb_dim % num_heads == 0
#         self.num_heads = num_heads
#         self.d_k = emb_dim // num_heads

#         self.qkv = nn.Linear(emb_dim, emb_dim * 3)
#         self.out = nn.Linear(emb_dim, emb_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(emb_dim)

#     def forward(self, x):
#         B, T, D = x.size()
#         qkv = self.qkv(x)  # (B, T, 3D)
#         q, k, v = qkv.chunk(3, dim=-1)

#         # (B, T, H, d_k) -> (B, H, T, d_k)
#         q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
#         k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
#         v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

#         scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, H, T, T)
#         attn = torch.softmax(scores, dim=-1)
#         attn = self.dropout(attn)

#         out = (attn @ v)  # (B, H, T, d_k)
#         out = out.transpose(1, 2).contiguous().view(B, T, D)
#         out = self.out(out)
#         return self.norm(x + self.dropout(out))



# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, emb_dim, num_heads, dropout=0.1):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm = nn.LayerNorm(emb_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         attn_output, _ = self.attn(x, x, x)
#         x = self.norm(x + self.dropout(attn_output))
#         return x