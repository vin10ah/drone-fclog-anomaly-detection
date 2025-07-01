import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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

class FTTransformer2(nn.Module):
    def __init__(self, num_numerical_features, d_token=64, n_blocks=4, n_heads=4, dropout=0.2, n_classes=1):
        super().__init__()
        
        self.d_token = d_token
        self.num_numerical_features = num_numerical_features
        
        # 간단한 토크나이저 (Sequential 피하기)
        self.input_proj = nn.Linear(num_numerical_features, d_token)
        self.input_norm = nn.LayerNorm(d_token)
        self.input_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            CompilerFriendlyTransformerBlock(d_token, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        
        # 출력 헤드 (Sequential 피하기)
        self.output_norm = nn.LayerNorm(d_token)
        self.output_linear1 = nn.Linear(d_token, d_token)
        self.output_linear2 = nn.Linear(d_token, n_classes)
        self.output_dropout = nn.Dropout(dropout)
        
    def forward(self, x_num):
        # 토크나이저
        x = self.input_proj(x_num)  # (B, d_token)
        x = self.input_norm(x)
        x = F.gelu(x)
        x = self.input_dropout(x)
        
        # 고정된 시퀀스 길이로 변환 (unsqueeze 대신 view 사용)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.d_token)  # (B, 1, d_token)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 시퀀스 차원 제거 (mean 대신 직접 인덱싱)
        x = x[:, 0, :]  # (B, d_token) - 첫 번째 토큰만 사용
        
        # 출력 헤드
        x = self.output_norm(x)
        x = self.output_linear1(x)
        x = F.gelu(x)
        x = self.output_dropout(x)
        x = self.output_linear2(x)
        
        return x