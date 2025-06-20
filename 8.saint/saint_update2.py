import torch
import torch.nn as nn

class EinsumSelfAttention(nn.Module):
    """
    NPU-safe Multi-Head Self-Attention with einsum:
    - No bmm
    - No permute
    - No chunk
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
        B, T, D = x.shape
        H = self.num_heads
        d_k = self.head_dim

        # (B, T, D) -> (B, T, H, d_k)
        q = self.q_proj(x).view(B, T, H, d_k)
        k = self.k_proj(x).view(B, T, H, d_k)
        v = self.v_proj(x).view(B, T, H, d_k)

        # Scaled dot-product Attention with einsum
        # scores: (B, H, T, T)
        scores = torch.einsum("bthd,bshd->bhts", q, k) / (d_k ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Context: (B, T, H, d_k)
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        out = out.reshape(B, T, D)

        out = self.out_proj(out)
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
    SAINT Block with einsum Attention:
    Column-wise Attn -> FFN -> Row-wise Attn -> FFN
    """
    def __init__(self, emb_dim, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.col_attn = EinsumSelfAttention(emb_dim, num_heads, dropout)
        self.ffn1 = FeedForward(emb_dim, dim_ff, dropout)
        self.row_attn = EinsumSelfAttention(emb_dim, num_heads, dropout)
        self.ffn2 = FeedForward(emb_dim, dim_ff, dropout)

    def forward(self, x):
        # Column-wise Attn
        x = self.col_attn(x)
        x = self.ffn1(x)

        # Row-wise Attn: treat (B, F, D) as (F, B, D)
        # einsum handles batch swapping internally
        # So we swap dims by reordering indices in einsum
        x_row = x.transpose(0, 1)  # (F, B, D)
        x_row = self.row_attn(x_row)
        x = x_row.transpose(0, 1)  # (B, F, D)
        x = self.ffn2(x)
        return x


class SAINT(nn.Module):
    """
    NPU-safe SAINT with einsum Attention
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
