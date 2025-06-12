import torch
import torch.nn as nn

class SAINTModel(nn.Module):
    def __init__(self, num_categories=None, num_numericals=0, emb_dim=32, num_heads=4, num_classes=2):
        super().__init__()
        self.has_cat = bool(num_categories)
        self.has_num = num_numericals > 0
        self.emb_dim = emb_dim

        # 범주형 feature embedding
        if self.has_cat:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(num_cat, emb_dim) for num_cat in num_categories
            ])

        # 수치형 feature embedding
        if self.has_num:
            self.num_linear = nn.Linear(num_numericals, emb_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classification head
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x_cat=None, x_num=None):
        batch_size = x_cat.size(0) if x_cat is not None else x_num.size(0)
        tokens = []

        if self.has_cat and x_cat is not None:
            cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            cat_embeds = torch.stack(cat_embeds, dim=1)  # shape: (B, N_cat, D)
            tokens.append(cat_embeds)

        if self.has_num and x_num is not None:
            num_embed = self.num_linear(x_num).unsqueeze(1)  # shape: (B, 1, D)
            tokens.append(num_embed)

        # concat embeddings
        x = torch.cat(tokens, dim=1)  # shape: (B, N_total, D)

        # pass through transformer
        x = self.transformer(x)  # shape: (B, N_total, D)

        # mean pooling
        x = x.mean(dim=1)  # shape: (B, D)

        return self.classifier(x)  # shape: (B, num_classes)
