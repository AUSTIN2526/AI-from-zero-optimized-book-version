import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPE(nn.Module):
    def __init__(self, d_model, max_seq_len=8192, base=10000):
        super(RoPE, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 預計算頻率值 (frequencies)
        # inv_freq = 1 / (base^(2i/d_model)) for i in [0, 1, ..., d_model//2-1]
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 預計算 cos 和 sin 表，避免重複計算
        self._precompute_freqs(max_seq_len)
    
    def _precompute_freqs(self, seq_len):
        # 位置索引 [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=torch.float32)
        
        # 計算每個位置和每個頻率的乘積
        # freqs shape: (seq_len, d_model//2)
        freqs = torch.outer(t, self.inv_freq)
        
        # 將頻率重複兩次以匹配 d_model 維度
        # emb shape: (seq_len, d_model)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 預計算 cos 和 sin
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x, seq_len=None, start_pos=0):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # 如果序列長度超過預計算的長度，重新計算
        if start_pos + seq_len > self.cos_cached.shape[0]:
            self._precompute_freqs(start_pos + seq_len)
        
        # 取得對應位置的 cos 和 sin 值
        cos = self.cos_cached[start_pos:start_pos + seq_len]
        sin = self.sin_cached[start_pos:start_pos + seq_len]
        
        # 擴展維度以匹配輸入張量的形狀
        # cos, sin: (seq_len, d_model) -> (..., seq_len, d_model)
        cos = cos.unsqueeze(0).expand_as(x)
        sin = sin.unsqueeze(0).expand_as(x)
        
        # 應用旋轉變換: x_rotated = x * cos + rotate_half(x) * sin
        return x * cos + self._rotate_half(x) * sin


class GatedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1, activation = 'relu'):
        super(GatedFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff * 2)  # 輸出兩倍維度，作為 GLU 輸入
        self.linear2 = nn.Linear(d_ff, d_model)      # 第二次線性變換，降回原始維度
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'silu' or activation == 'swish':
            self.activation = F.silu
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError('Unsupported activation function: {}'.format(activation))
    
    def forward(self, x):
        # 將輸出切成兩半：一部分作為輸出，一部分作為 gate
        x_proj = self.linear1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x_gated = x1 * self.activation(x2)  # 套用 gating 機制
        return self.linear2(self.dropout(x_gated))


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, nhead, num_kv_heads, dropout=0.1, max_seq_len=8192):
        super(GroupedQueryAttention, self).__init__()
        assert d_model % nhead == 0, "d_model 必須可以整除 nhead"
        assert nhead % num_kv_heads == 0, "nhead 必須可以整除 num_kv_heads"
        
        self.d_model = d_model
        self.nhead = nhead  # query heads 數量
        self.num_kv_heads = num_kv_heads  # key-value heads 數量
        self.d_k = d_model // nhead  # 每個 head 的維度大小
        self.num_queries_per_kv = nhead // num_kv_heads  # 每個 KV head 對應的 Q head 數量
        
        # Query 仍然使用全部的 heads
        self.w_q = nn.Linear(d_model, d_model)
        # Key 和 Value 只使用較少的 heads
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_o = nn.Linear(d_model, d_model)  # 最終輸出映射層
        
        # 為每個 head 建立 RoPE
        self.rope = RoPE(self.d_k, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)  # 縮放因子，用於穩定 softmax
    
    def apply_rope_to_heads(self, x, seq_len, start_pos=0):
        """將 RoPE 應用到多頭張量上"""
        batch_size, num_heads, seq_len_actual, d_k = x.shape
        
        # 使用 contiguous() 確保記憶體連續，然後重塑為 (batch_size * num_heads, seq_len, d_k)
        x_reshaped = x.contiguous().view(batch_size * num_heads, seq_len_actual, d_k)
        
        # 應用 RoPE
        x_rope = self.rope(x_reshaped, seq_len, start_pos)
        
        # 重塑回 (batch_size, num_heads, seq_len, d_k)
        return x_rope.view(batch_size, num_heads, seq_len_actual, d_k)
    
    def forward(self, query, key, value, mask=None, key_padding_mask=None, start_pos=0):
        batch_size, seq_len, _ = query.size()
        kv_seq_len = key.size(1)
        
        # Query: (batch_size, nhead, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Key 和 Value: (batch_size, num_kv_heads, kv_seq_len, d_k)
        K = self.w_k(key).view(batch_size, kv_seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, kv_seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 對 Q 和 K 應用 RoPE (V 不需要位置編碼)
        Q = self.apply_rope_to_heads(Q, seq_len, start_pos)
        K = self.apply_rope_to_heads(K, kv_seq_len, start_pos)
        
        # 將 K 和 V 重複擴展以匹配 Q 的 head 數量
        # 使用 repeat_interleave 讓每個 KV head 對應多個 Q head
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)  # (batch_size, nhead, kv_seq_len, d_k)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)  # (batch_size, nhead, kv_seq_len, d_k)
        
        # 計算注意力分數
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 套用 attention mask（如 causal mask）
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # 套用 padding mask，忽略 padding 位置
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        # 計算注意力權重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 計算 weighted sum 後 reshape 回原始形狀
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 最終輸出線性變換
        output = self.w_o(context)
        return output