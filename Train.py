import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformer import Transformer
from ProcessData import ImgDataset, collate_fn

class MolTranslateModel:
    def __init__(self,vocab_size, map_size=32, dim=32,
                 n_head=8, dropout=0.3, decoder_layers=3,
                 batchsize=8):

        self.device = "cuda:0"
        self.pad_index = 0

        self.scaler = GradScaler('cuda')
        self.model = Transformer(
            vocab_size, map_size, dim,
            n_head, dropout, decoder_layers
        ).to(self.device)
        self.batchsize = batchsize

        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.pad_index)

    def train(self, epoch):
        train_data_path = "F:/aitificial_intelligence/data/mol_img/train_data"
        train_data_idx = 1
        train_dataset = ImgDataset.load_from_tensor_file(
            f"{train_data_path}_{train_data_idx}.pt")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batchsize,
            shuffle=True,
            collate_fn=collate_fn
        )
        """
        test_data_path = "F:/aitificial_intelligence/data/mol_img/train_data"
        test_data_idx = 2
        test_dataset = ImgDataset.load_from_tensor_file(
            f"{test_data_path}_{test_data_idx}.pt")

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batchsize,
            shuffle=True,
            collate_fn=collate_fn
        )
        """


        for idx in range(epoch):
            self.model.train()
            total_loss = 0.
            for img, tgt in train_dataloader:
                img = img.to(self.device)
                tgt = tgt.to(self.device)

                self.optimizer.zero_grad()

                with autocast('cuda'):
                    output = self.model(img, tgt)
                    output_logit = output.view(-1, output.size(-1))
                    print(output)
                    print(torch.max(output))
                    tgt_flatten = tgt.view(-1)
                    loss = self.loss_function(output_logit, tgt_flatten)

                #self.scaler.scale(loss).backward()
                #self.scaler.unscale_(self.optimizer)
                loss.backward()
                #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                #self.scaler.step(self.optimizer)
                #self.scaler.update()

                total_loss += loss.item()
                break
            print(f"Epoch {idx} finished")
            print(f"Total loss: {total_loss}")

            """
            self.model.eval()
            total_loss = 0.
            # correct = 0
            for img, tgt in test_dataloader:
                img = img.to(self.device)
                tgt = tgt.to(self.device)

                output = self.model(img, tgt)
                output_logit = output.view(-1, output.size(-1))
                tgt_flatten = tgt.view(-1)
                loss = self.loss_function(output_logit, tgt_flatten)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                # correct += (pred == tgt).sum().item()
                # total += tgt.size(0)
            print(f"validate total loss: {total_loss}")
            """





if __name__ == "__main__":
    import pickle as pkl
    from Vocabulary import Vocabulary
    with open("vocabulary_dictionary.pkl", "rb") as f:
        loaded_dict = pkl.load(f)
    vocab = Vocabulary.load_from_dictionary(loaded_dict)
    model = MolTranslateModel(len(vocab))
    model.train(1)








