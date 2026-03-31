import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from transformer import Transformer
from ProcessData import ImgDataset, collate_fn
from Vocabulary import Vocabulary

class MolTranslateModel:
    def __init__(self,vocab_size, map_size=32, dim=32,
                 n_head=8, dropout=0.3, decoder_layers=1,
                 batchsize=8, total_epoch=None):

        self.device = "cuda:0"
        self.total_epoch = total_epoch
        self.pad_index = 0

        self.vocab_size = vocab_size
        self.dropout = dropout

        self.scaler = GradScaler('cuda')
        self.model = Transformer(
            vocab_size, map_size, dim,
            n_head, dropout, decoder_layers
        ).to(self.device)
        self.batchsize = batchsize

        self.optimizer = None
        self.scheduler = None

        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.pad_index)

        self.dataloader = {}
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_acc': []
        }

    def load_data(self, file_idx_list, type: str="train"):
        data_prefix = "F:/aitificial_intelligence/data/mol_img/train_data"
        data_path_list = [f"{data_prefix}_{idx}.pt" for idx in file_idx_list]

        dataset = ImgDataset.load_from_tensor_files(data_path_list)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batchsize,
            shuffle=(True if type == "train" else False),
            collate_fn=collate_fn
        )
        self.dataloader[type] = dataloader
        return dataloader

    def set_learning_strategy(self, lr=1e-4, warmup_ratio=0.1):

        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(self.dataloader["train"]) * self.total_epoch
        warmup_steps = total_steps * warmup_ratio

        scheduler_warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        scheduler_decay = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps-warmup_steps,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler_warmup, scheduler_decay],
            milestones=[warmup_steps]
        )

    def check_nan(self, tensor, name=""):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains NaN/Inf!")

    def train_one_batch(self, epoch):
        dataloader = self.load_data([1])
        img, tgt = next(iter(dataloader))
        img = img.to(self.device)
        tgt = tgt.to(self.device).long()
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        self.model.train()
        for idx in range(epoch):
            self.optimizer.zero_grad()

            with autocast('cuda'):
                output = self.model(img, tgt_input)
                output_logit = output.reshape(-1, output.size(-1))

                tgt_flatten = tgt_label.reshape(-1)
                loss = self.loss_function(output_logit, tgt_flatten)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            print(f"loss: {loss.item()}")

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.
        for img, tgt in self.dataloader["train"]:
            img = img.to(self.device).float()
            tgt = tgt.to(self.device).long()
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            self.optimizer.zero_grad()

            with autocast('cuda'):
                output = self.model(img, tgt_input)
                output_logit = output.reshape(-1, output.size(-1))
                tgt_flatten = tgt_label.reshape(-1)
                loss = self.loss_function(output_logit, tgt_flatten)
                self.check_nan(loss, f"loss (epoch {epoch_idx})")

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.scheduler.step()
            total_loss += loss.item()

        average_loss = total_loss / len(self.dataloader['train'])
        self.history['train_loss'].append((epoch_idx, average_loss))
        print(f"########## train epoch {epoch_idx} ##########")
        print(f"Total loss: {total_loss :.4f}")
        print(f"Average loss: {average_loss :.4f}")

    @torch.no_grad()
    def validate(self, epoch_idx):
        self.model.eval()
        total_loss = 0.
        total_correct = 0
        total_tokens = 0

        for img, tgt in self.dataloader["validate"]:
            img = img.to(self.device).float()
            tgt = tgt.to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            output = self.model(img, tgt_input)

            output_logit = output.reshape(-1, output.size(-1))
            tgt_flatten = tgt_label.reshape(-1)

            loss = self.loss_function(output_logit, tgt_flatten)
            total_loss += loss.item()

            non_pad_mask = (tgt_label != self.pad_index)

            pred = output.argmax(dim=-1)
            total_correct += ((pred == tgt_label) & non_pad_mask).sum().item()
            total_tokens += non_pad_mask.sum().item()

        average_loss = total_loss / len(self.dataloader['validate'])
        accuracy = total_correct / total_tokens
        self.history['val_loss'].append((epoch_idx, average_loss))
        self.history['val_acc'].append((epoch_idx, accuracy))
        print(f"########## validate epoch {epoch_idx} ##########")
        print(f"validate total loss: {total_loss :.4f}")
        print(f"validate average loss: {average_loss :.4f}")
        print(f"Token accuracy: {accuracy :.4f}")

        return average_loss, accuracy

    def save_checkpoint(self, epoch, save_path="checkpoints"):
        import os
        os.makedirs(save_path, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
        }

        torch.save(checkpoint, f"{save_path}/checkpoint_epoch_{epoch}.pt")
        print(f"Checkpoint saved to {save_path}/checkpoint_epoch_{epoch}.pt")


    def load_checkpoint(self, checkpoint_path):

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")

        return checkpoint['epoch'], checkpoint['history']

    def save_model(self, save_path="models/final_model.pt"):
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        final_model = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': self.vocab_size,
                'map_size': self.model.encoder.cnn_layer.output_size,
                'dim': self.model.decoder_embedding.embedding_dim,
                'n_head': self.model.decoder_layers[0].self_attn.n_head,
                'decoder_layers': len(self.model.decoder_layers),
                'dropout': self.dropout
            },
        }
        torch.save(final_model, save_path)
        print(f"final model saved to {save_path}")
        return save_path

    @classmethod
    def load_for_inference(cls, model_path, device='cuda:0'):
        model_info = torch.load(model_path, map_location=device)
        config = model_info['model_config']

        model = cls(
            vocab_size=config['vocab_size'],
            map_size=config['map_size'],
            dim=config['dim'],
            n_head=config['n_head'],
            dropout=config['dropout'],
            decoder_layers=config['decoder_layers'],
            batchsize=1
        )

        model.model.load_state_dict(model_info['model_state_dict'])
        model.model.to(device)
        model.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Device: {device}")

        return model

    @torch.no_grad()
    def generate(self, img, vocb: Vocabulary):
        self.model.eval()

        decoder_input = torch.tensor([[vocb.bos_index]], device=self.device)
        generate_tokens = []

        for _ in range(100):
            output = self.model(img, decoder_input)
            next_token_logits = output[0, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            if next_token.item() == vocb.eos_index:
                break
            generate_tokens.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)

        return generate_tokens


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss -self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


if __name__ == "__main__":
    import pickle as pkl
    from Vocabulary import Vocabulary
    with open("vocabulary_dictionary.pkl", "rb") as f:
        loaded_dict = pkl.load(f)
    vocab = Vocabulary.load_from_dictionary(loaded_dict)
    total_epoch = 30
    model = MolTranslateModel(
        len(vocab),
        total_epoch=30,
        map_size=16,
        dim=32,
        n_head=4,
        dropout=0.1,
        decoder_layers=1,
        batchsize=16
    )
    model.load_data([1], "train")
    model.load_data([2], "validate")
    model.set_learning_strategy()
    best_val_loss = float('inf')
    best_val_acc = 0.

    early_stopping = EarlyStopping(patience=5)
    for epoch in range(1, total_epoch+1):
        model.train_one_epoch(epoch)

        if epoch % 5 == 0:
            average_loss, accuracy = model.validate(epoch)
            if average_loss < best_val_loss and accuracy > best_val_acc:
                model.save_checkpoint(epoch)
            if early_stopping(average_loss):
                print("Early stopping triggered")
                break
    print("train finished")











