import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from SelfAttention import MutiHeadAttention
from PositionEncoding import PositionalEncoding1D, PositionalEncoding2D
from CNN import CNN


class EncoderLayer(nn.Module):
    def __init__(self, dim, n_head, dropout=0.1):
        super().__init__()
        
        self.self_attn = MutiHeadAttention(n_head, dim, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self Attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed Forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, n_head, dropout=0.1):
        super().__init__()
        
        self.self_attn = MutiHeadAttention(n_head, dim, dropout)
        self.cross_attn = MutiHeadAttention(n_head, dim, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self Attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross Attention
        cross_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, map_size=32, dim=32,
                 n_head=8, dropout=0.1, decoder_layers=3):
        super().__init__()
        
        self.dim = dim
        self.map_size = map_size
        
        # Image encoder (CNN)
        self.cnn = CNN(hidden_channel=32, output_channel=dim, output_size=map_size)
        
        # Flatten CNN output: [batch, dim, H, W] -> [batch, H*W, dim]
        self.img_proj = nn.Linear(map_size * map_size, dim)
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        
        # Position encoding for sequence
        self.pos_encoder = PositionalEncoding1D()
        
        # Encoder layers (using CNN features directly, can add transformer encoder if needed)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim, n_head, dropout) for _ in range(decoder_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization to prevent NaN"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Small initialization for embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, img, tgt):
        """
        img: [batch, 1, H, W] - image tensor
        tgt: [batch, seq_len] - target token indices
        """
        batch_size = img.size(0)
        tgt_seq_len = tgt.size(1)
        
        # Image encoding
        img_feat = self.cnn(img)  # [batch, dim, H, W]
        img_feat = img_feat.flatten(2)  # [batch, dim, H*W]
        img_feat = img_feat.transpose(1, 2)  # [batch, H*W, dim]
        
        # Token embedding
        tgt_embed = self.token_embedding(tgt)  # [batch, seq_len, dim]
        tgt_embed = tgt_embed * np.sqrt(self.dim)  # Scale embedding
        
        # Add position encoding
        tgt_embed = self.pos_encoder(tgt_embed)  # [batch, seq_len, dim]
        
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(img.device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Pass through decoder layers
        x = tgt_embed
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, img_feat, src_mask=None, tgt_mask=tgt_mask)
        
        # Output projection
        output = self.output_proj(x)  # [batch, seq_len, vocab_size]
        
        return output
