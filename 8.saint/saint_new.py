import torch
import torch.nn as nn

class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = emb_dim // num_heads

        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.out = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        B, T, D = x.size()
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, H, d_k) -> (B, H, T, d_k)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, H, T, T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        return self.norm(x + self.dropout(out))



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x

class FeedForward(nn.Module):
    def __init__(self, emb_dim, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class SAINTBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.col_attn = CustomMultiHeadSelfAttention(emb_dim, num_heads, dropout)
        self.ffn1 = FeedForward(emb_dim, dim_feedforward, dropout)
        self.row_attn = CustomMultiHeadSelfAttention(emb_dim, num_heads, dropout)
        self.ffn2 = FeedForward(emb_dim, dim_feedforward, dropout)

    def forward(self, x):
        # Column-wise attention
        x = self.col_attn(x)
        x = self.ffn1(x)

        # Row-wise attention: swap batch & feature dims
        x = x.transpose(0, 1)  # (F, B, D)
        x = self.row_attn(x)
        x = x.transpose(0, 1)  # (B, F, D)

        x = self.ffn2(x)
        return x

class SAINT(nn.Module):
    def __init__(self, num_categories=None, num_numericals=0, emb_dim=32, num_heads=4,
                 num_classes=1, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.has_cat = bool(num_categories)
        self.has_num = num_numericals > 0
        self.emb_dim = emb_dim

        # Categorical feature embeddings
        if self.has_cat:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(num_cat, emb_dim) for num_cat in num_categories
            ])

        # Numerical feature projection
        if self.has_num:
            self.num_linear = nn.Linear(num_numericals, emb_dim)

        # SAINT Blocks (Col+Row Attention)
        self.blocks = nn.ModuleList([
            SAINTBlock(emb_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Classification head
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

        # Combine
        x = torch.cat(tokens, dim=1)  # (B, F, D)

        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, D)

        return self.classifier(x)  # (B, num_classes)
