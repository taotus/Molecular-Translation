import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, dim, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.dim_head = dim // n_head

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        :param x: [batch_size, seq_length, dim]
        :return: [batch_size, n_heads, seq_length, dim_head]
        """
        batch_size, seq_len, dim = x.size()
        return x.view(batch_size, seq_len, self.n_head, self.dim_head).transpose(1, 2)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        :param Q K V: [batch_size, n_heads, seq_length, dim_head]
        :param mask:
        :return:
        """
        # attention_scores: [batch_size, n_heads, seq_length, seq_length]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.dim_head)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == True, -torch.inf)

        attn_probs = torch.softmax(attention_scores, dim=-1) #不一定要softmax

        output = torch.matmul(attn_probs, V) #形状同Q
        return output

    def merge_heads(self, x):
        """
        :param x: [batch_size, n_heads, seq_length, dim_head]
        :return: [batch_size, seq_length, dim]
        """
        batch_size, n_heads, seq_len, dim_head = x.size()
        return x.transpose(1, 2).reshape(batch_size, seq_len, -1)


    def forward(self, query, key, value, mask=None):
        """
        query: [batch_size, seq_len_q, dim]
        key:   [batch_size, seq_len_k, dim]
        value: [batch_size, seq_len_v, dim]
        mask:  [batch_size, seq_len_q, seq_len_k]
        """
        if query.isnan().any() or key.isnan().any() or value.isnan().any():
            print("query/key/value 包含 NaN")
        Q = self.split_heads(self.w_q(query))
        K = self.split_heads(self.w_k(key))
        V = self.split_heads(self.w_v(value))
        if Q.isnan().any() or K.isnan().any() or V.isnan().any():
            print("Q/K/V 包含 NaN")

        #attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        #attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        if attn_output.isnan().any() or attn_output.isinf().any():
            print(f"attn_output 包含 NaN/Inf")
            print(f"Q stats: min={Q.min()}, max={Q.max()}")
            print(f"K stats: min={K.min()}, max={K.max()}")

        output = self.w_o(self.merge_heads(attn_output))

        return output
