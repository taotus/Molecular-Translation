import torch
import torch.nn as nn
import pickle as pkl


from CNN import CNN
from PositionEncoding import PositionalEncoding2D, PositionalEncoding1D
from SelfAttention import MultiHeadAttention
from Vocabulary import Vocabulary, Tokenizer, SMILES_PATTERN

class FeedForwardNet(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.fc_layer1 = nn.Linear(dim, hidden_dim)
        self.fc_layer2 = nn.Linear(hidden_dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc_layer2(self.fc_layer1(x)))

class Encoder(nn.Module):
    def __init__(self, dim=32, output_size=32):
        super().__init__()

        self.cnn_layer = CNN(
            dim,
            dim,
            output_size
        )
        self.feed_forward = FeedForwardNet(
            dim, 2 * dim
        )
        self.positional_encoding_layer = PositionalEncoding2D()

    def forward(self, img):
        feature_map = self.cnn_layer(img).permute(0, 2, 3, 1)
        batch = feature_map.shape[0]
        dim = feature_map.shape[-1]
        positional_seq = self.positional_encoding_layer(feature_map).view(batch, -1, dim)
        encoded = self.feed_forward(positional_seq)
        return encoded

class DecodeLayer(nn.Module):
    def __init__(self, dim=32, n_head=8, dropout=0.3):
        super().__init__()

        self.masked_muti_head_attention = MultiHeadAttention(
            n_head, dim, dropout
        )
        self.norm_layer1 = nn.LayerNorm(dim)
        self.norm_layer2 = nn.LayerNorm(dim)
        self.norm_layer3 = nn.LayerNorm(dim)

        self.self_attn = MultiHeadAttention(n_head, dim, dropout)
        self.cross_attn = MultiHeadAttention(n_head, dim, dropout)
        self.feed_forward = FeedForwardNet(dim, 2 * dim)

    def forward(self, x, encoder_output, target_mask):

        x = self.norm_layer1(x)
        x = x + self.self_attn(x, x, x, target_mask)

        x = self.norm_layer2(x)
        x = x + self.cross_attn(x, encoder_output, encoder_output)

        x = self.norm_layer3(x)
        x = x + self.feed_forward(x)

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, map_size=32, dim=32, n_head=8, dropout=0.3,
                 decoder_layers=3):
        super().__init__()

        self.decoder_embedding = nn.Embedding(vocab_size, dim)
        self.positional_encoding_1d = PositionalEncoding1D()

        self.encoder = Encoder(
            dim, map_size
        )
        self.decoder_layers = nn.ModuleList(
            [DecodeLayer(dim, n_head, dropout) for _ in range(decoder_layers)]
        )
        self.fc = nn.Linear(dim, vocab_size)

    def generate_target_mask(self, target, pad_idx):
        """
        :param target: [B, seq_len]
        :return: mask: [B, 1, seq_len, seq_len]
        """
        batch_size, target_length = target.size()
        device = target.device

        look_ahead_mask = torch.triu(torch.ones(target_length, target_length, device=device), diagonal=1).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)

        padding_mask = (target == pad_idx).unsqueeze(1).unsqueeze(2) # [B, 1, 1, L]

        tgt_mask = look_ahead_mask | padding_mask
        #print(f"mask 中 True 比例：{tgt_mask.float().mean():.4f}")  # True=屏蔽，应该有合理比例
        row_all_masked = tgt_mask.all(dim=-1)
        if row_all_masked.any():
            print("row all masked")
        return tgt_mask

    def forward(self, img, tgt_seq, pad_idx=0):
        tgt_embedding = self.decoder_embedding(tgt_seq)
        if tgt_embedding.isnan().any():
            print("tgt embedding have nan")
        #print(f"tgt embedding: {tgt_embedding.shape}")
        tgt_mask = self.generate_target_mask(tgt_seq, pad_idx)
        encoder_output = self.encoder(img)
        if encoder_output.isnan().any():
            print("encoder output have nan")

        decoder_output = self.positional_encoding_1d(tgt_embedding)
        if decoder_output.isnan().any():
            print("pe have nan")
        #print(f"decoder input shape: {decoder_output.shape}")
        for decode_layer in self.decoder_layers:
            decoder_output = decode_layer(decoder_output, encoder_output, tgt_mask)
            if decoder_output.isnan().any():
                print("decoder output have nan")
        #print(f"decoder output shape: {decoder_output.shape}")
        output = self.fc(decoder_output)
        #print(f"output shape: {output.shape}")
        return output

if __name__ == "__main__":
    batch_smiles = [
        "C[C@@]12[C@@]([H])([C@]3([C@@H](CC2)OC(OC3)C4=C(C=CC(Br)=C4)O)C)CCC([C@H]1C/C=C5C(OC[C@H]/5O)=O)=C",
        "C[C@@]12[C@@]([H])([C@]3([C@@H](CC2)OC(OC3)C4=CC=C(C=C4)[N+]([O-])=O)C)CCC([C@H]1C/C=C5C(OC[C@H]/5O)=O)=C",
        "C[C@@]12[C@@]([H])([C@]3([C@@H](CC2)OC(OC3)C4=CC=C(C=C4Cl)F)C)CCC([C@H]1C/C=C5C(OC[C@H]/5O)=O)=C"
    ]
    with open("vocabulary_dictionary.pkl", "rb") as f:
        loaded_dict = pkl.load(f)
    vocab = Vocabulary.load_from_dictionary(loaded_dict)

    tokenizer = Tokenizer(pattern=SMILES_PATTERN, vocab=vocab)
    tokenized_smiles = tokenizer.batch_tokenize(batch_smiles, pad=True)

    target = torch.tensor(tokenized_smiles)
    print(f"target shape: {target.shape}")

    img_ids = [
        "000a1df30dfe",
        "000a2d0c1855",
        "000a2d1f9cb8"
    ]
    from ImgStandardize import process_images

    mean = 251.7802
    std = 28.4787

    img_tensor = process_images(
        img_ids, mean, std, 0, False
    ).float()
    print(f"image shape: {img_tensor.shape}")
    transformer = Transformer(
        vocab_size=len(vocab),
    )
    output = transformer(
        img_tensor, target
    )
    print(f"output shape: {output.shape}")






















