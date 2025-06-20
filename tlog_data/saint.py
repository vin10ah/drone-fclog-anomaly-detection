import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))

class SAINTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.col_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn1 = FeedForward(d_model, dim_feedforward, dropout)
        self.row_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn2 = FeedForward(d_model, dim_feedforward, dropout)

    def forward(self, x):
        # x: (B, F, D)
        x = self.col_attn(x)     # Column-wise attention
        x = self.ffn1(x)
        x = x.transpose(1, 2)    # For row-wise attention
        x = self.row_attn(x)
        x = self.ffn2(x)
        x = x.transpose(1, 2)
        return x

class SAINT(nn.Module):
    def __init__(self, num_categories, category_dims, num_continuous, d_model=64, n_blocks=2, n_heads=4, dim_feedforward=256, dropout=0.1, n_classes=1):
        super().__init__()
        # Embedding for each categorical feature
        self.cat_embeds = nn.ModuleList([
            nn.Embedding(cat_size, d_model) for cat_size in category_dims
        ])
        self.continuous_proj = nn.Linear(num_continuous, d_model)

        self.blocks = nn.ModuleList([
            SAINTBlock(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_blocks)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x_cat, x_cont):
        # Categorical embedding
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)]
        x_cat = torch.stack(cat_embeds, dim=1)  # (B, num_cat, D)

        # Continuous projection
        x_cont = self.continuous_proj(x_cont).unsqueeze(1)  # (B, 1, D)

        # Combine
        x = torch.cat([x_cat, x_cont], dim=1)  # (B, F, D)

        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, D)
        return self.classifier(x)
