import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PositionalEncoding1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, max_seq_len, dim = x.size()
        device = x.device

        PE = torch.zeros([max_seq_len, dim], device=device)
        position = torch.arange(0, max_seq_len, device=device).unsqueeze(1) # [max_seq_len, 1]
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() *
                    (-torch.log(torch.tensor(10000.0, device=device)) / dim))

        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('PE', PE)

        return x + PE

class PositionalEncoding2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, H, W, dim = x.size()
        device = x.device

        d_h = dim // 2
        d_w = dim - d_h

        pos_h = torch.arange(0, H, device=device).unsqueeze(1) # [H, 1]
        div_term_h = torch.exp(torch.arange(0, d_h, 2, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / d_h))
        pe_h = torch.zeros([H, d_h], device=device)
        pe_h[:, 0::2] = torch.sin(pos_h * div_term_h)
        pe_h[:, 1::2] = torch.cos(pos_h * div_term_h)

        pos_w = torch.arange(0, W, device=device).unsqueeze(1)
        pe_w = torch.zeros([W, d_h], device=device)
        div_term_w = torch.exp(torch.arange(0, d_w, 2, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / d_w))
        pe_w[:, 0::2] = torch.sin(pos_w * div_term_w)
        pe_w[:, 1::2] = torch.cos(pos_w * div_term_w)

        PE = torch.zeros([H, W, dim], device=device)
        PE[:, :, :d_h] = pe_h.unsqueeze(1).expand(H, W, d_h)
        PE[:, :, d_h:] = pe_w.unsqueeze(0).expand(H, W, d_w)

        return x + PE