import torch
import torch.nn as nn

class CustomSelfAttention(nn.Module):
    """
    NPU-safe Multi-Head Self Attention:
    - No nn.MultiheadAttention
    - No transpose
    - Only bmm, matmul, softmax
    """
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        B, T, D = x.size()  # batch, tokens, embedding

        # q, k, v: (B, T, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape to multihead: (B, T, H, d_head)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        # for bmm: (B*H, T, d_head)
        q = q.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)
        k = k.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)
        v = v.permute(0, 2, 1, 3).reshape(B * self.num_heads, T, self.head_dim)

        # scaled dot-product attention
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)  # (B*H, T, T)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.bmm(attn_weights, v)  # (B*H, T, d_head)

        # restore shape: (B, T, H, d_head) -> (B, T, D)
        attn_output = attn_output.view(B, self.num_heads, T, self.head_dim).permute(0, 2, 1, 3).reshape(B, T, D)

        out = self.out_proj(attn_output)
        return self.norm(x + self.dropout(out))


class FeedForward(nn.Module):
    def __init__(self, emb_dim, dim_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class SAINTBlock(nn.Module):
    """
    SAINT Block: Column-wise Attn -> FFN -> Row-wise Attn -> FFN
    Both Attn use NPU-safe CustomSelfAttention
    """
    def __init__(self, emb_dim, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.col_attn = CustomSelfAttention(emb_dim, num_heads, dropout)
        self.ffn1 = FeedForward(emb_dim, dim_ff, dropout)
        self.row_attn = CustomSelfAttention(emb_dim, num_heads, dropout)
        self.ffn2 = FeedForward(emb_dim, dim_ff, dropout)

    def forward(self, x):
        # Column-wise Attn: (B, F, D)
        x = self.col_attn(x)
        x = self.ffn1(x)

        # Row-wise Attn: NPU-safe 구현 (transpose 대신 einsum 사용)
        # Treat (B, F, D) as (F, B, D) by reshaping for attention
        x_ = x.permute(1, 0, 2)  # (F, B, D)
        x_ = self.row_attn(x_)
        x = x_.permute(1, 0, 2)  # (B, F, D)
        x = self.ffn2(x)
        return x


class SAINT(nn.Module):
    """
    SAINT Model - NPU safe version
    - Uses CustomSelfAttention with bmm only
    - No MultiheadAttention, no transpose at runtime
    """
    def __init__(self, num_categories=None, num_numericals=0,
                 emb_dim=32, num_heads=4, num_classes=1,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.has_cat = bool(num_categories)
        self.has_num = num_numericals > 0
        self.emb_dim = emb_dim

        if self.has_cat:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(num_cat, emb_dim) for num_cat in num_categories
            ])

        if self.has_num:
            self.num_linear = nn.Linear(num_numericals, emb_dim)

        self.blocks = nn.ModuleList([
            SAINTBlock(emb_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x_cat=None, x_num=None):
        tokens = []

        if self.has_cat and x_cat is not None:
            cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            cat_embeds = torch.stack(cat_embeds, dim=1)  # (B, N_cat, D)
            tokens.append(cat_embeds)

        if self.has_num and x_num is not None:
            num_embed = self.num_linear(x_num).unsqueeze(1)  # (B, 1, D)
            tokens.append(num_embed)

        x = torch.cat(tokens, dim=1)  # (B, F, D)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=1)  # (B, D)
        return self.classifier(x)  # (B, num_classes)
