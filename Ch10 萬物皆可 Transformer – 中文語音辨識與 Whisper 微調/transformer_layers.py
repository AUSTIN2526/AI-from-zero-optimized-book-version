import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 修正：調整為 (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # 統一處理，支援 (batch_size, seq_len, d_model) 格式
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # 將 Q/K/V 映射後 reshape 成 (batch_size, nhead, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, seq_len_q, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.nhead, self.d_k).transpose(1, 2)
        
        # 計算注意力分數
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 套用 attention mask
        if mask is not None:
            # 確保 mask 的形狀正確
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 套用 padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, seq_len_k)
            # 需要擴展為 (batch_size, 1, 1, seq_len_k)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
        # 計算注意力權重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 計算 weighted sum 後 reshape 回原始形狀
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # 最終輸出線性變換
        output = self.w_o(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError('Unsupported activation function: {}'.format(activation))
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1, norm=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadAttention(d_model, nhead, dropout),
                'feed_forward': FeedForward(d_model, d_ff, dropout),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            # 自注意力機制 + 殘差連接 + LayerNorm
            src2 = layer['self_attn'](output, output, output, mask, src_key_padding_mask)
            output = layer['norm1'](output + layer['dropout'](src2))

            # 前饋神經網路 + 殘差連接 + LayerNorm
            src2 = layer['feed_forward'](output)
            output = layer['norm2'](output + layer['dropout'](src2))

        if self.norm is not None:
            output = self.norm(output)
        return output


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, dropout=0.1, norm=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': MultiHeadAttention(d_model, nhead, dropout),
                'cross_attn': MultiHeadAttention(d_model, nhead, dropout),
                'feed_forward': FeedForward(d_model, d_ff, dropout),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            # 自注意力機制（含遮罩）+ 殘差連接 + LayerNorm
            tgt2 = layer['self_attn'](output, output, output, tgt_mask, tgt_key_padding_mask)
            output = layer['norm1'](output + layer['dropout'](tgt2))
            
            # Cross attention（如果有 memory）
            if memory is not None:
                tgt2 = layer['cross_attn'](output, memory, memory, memory_mask, memory_key_padding_mask)
                output = layer['norm2'](output + layer['dropout'](tgt2))
            
            # 前饋神經網路 + 殘差連接 + LayerNorm
            tgt2 = layer['feed_forward'](output)
            output = layer['norm3'](output + layer['dropout'](tgt2))
        
        if self.norm is not None:
            output = self.norm(output)
        return output


# 用於測試的輔助函數
def create_causal_mask(seq_len, device=None):
    """創建因果遮罩（下三角矩陣）"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    if device is not None:
        mask = mask.to(device)
    return mask


def create_padding_mask(seq_lengths, max_len):
    """創建 padding 遮罩"""
    batch_size = len(seq_lengths)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        if length < max_len:
            mask[i, length:] = True
    return mask