import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle as pkl

from ImgStandardize import image2tensor, pad_images
from Vocabulary import Vocabulary, Tokenizer, SMILES_PATTERN


class ImgDataset(Dataset):
    def __init__(self, train: bool=True):
        super().__init__()
        self.img_label_list = []
        self.train = train

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, idx):
        return self.img_label_list[idx]

    def build_from_csv(self, csv_path, chunk_size: int=None):

        if chunk_size is not None:
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                smiles_list = chunk["smiles"].tolist()
                img_id_list = chunk["image_id"].tolist()
                img_tensor_list = [image2tensor(idx, self.train) for idx in img_id_list]
                self.img_label_list.extend(zip(img_tensor_list, smiles_list))
                break
        else:
            df = pd.read_csv(csv_path)
            smiles_list = df["smiles"].tolist()
            img_id_list = df["image_id"].tolist()
            img_tensor_list = [image2tensor(idx, self.train) for idx in img_id_list]

            self.img_label_list = list(zip(img_tensor_list, smiles_list))

        print(f"Build ImgDataset csv file: {csv_path}")

    def build_from_df(self, df: pd.DataFrame):
        smiles_list = df["smiles"].tolist()
        img_id_list = df["image_id"].tolist()
        img_tensor_list = [image2tensor(idx, self.train) for idx in img_id_list]
        self.img_label_list = list(zip(img_tensor_list, smiles_list))

    def export_tensor_file(self, file_path):
        torch.save(
            self.img_label_list,
            file_path,
            _use_new_zipfile_serialization=True
        )
        print(f"ImgDataset saved to {file_path}")

    @classmethod
    def load_from_tensor_file(cls, file_path):
        img_dataset = cls()
        img_dataset.img_label_list = torch.load(file_path)
        print(f"ImgDataset load from {file_path}")
        return img_dataset

    @classmethod
    def load_from_tensor_files(cls, file_path_list):
        img_dataset = cls()
        img_dataset.img_label_list.clear()
        for path in file_path_list:
            img_dataset.img_label_list.extend(torch.load(path))
            print(f"Load from {path}")
        print(f"Load total {len(file_path_list)} tensor files")
        return img_dataset


def collate_fn(batch):

    img_list, tgt_list = zip(*batch)
    batch_imgs = pad_images(img_list, pad_value=0)

    with open("vocabulary_dictionary.pkl", "rb") as f:
        loaded_dict = pkl.load(f)
    vocab = Vocabulary.load_from_dictionary(loaded_dict)
    tk = Tokenizer(pattern=SMILES_PATTERN, vocab=vocab)

    batch_tgt = torch.tensor(tk.batch_tokenize(tgt_list), dtype=torch.long)

    return batch_imgs, batch_tgt

if __name__ == "__main__":
    csv_path = "mol_img/train_smiles.csv"

    for idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=50000)):
        img_dataset = ImgDataset(train=True)
        img_dataset.build_from_df(chunk)
        img_dataset.export_tensor_file(f"F:/aitificial_intelligence/data/mol_img/train_data_{idx+1}.pt")

    img_dataset = ImgDataset.load_from_tensor_files(["mol_img/train_data.pt"])
    dataloader = DataLoader(
        img_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )
    for data in dataloader:
        batch_img, batch_tgt = data
        print(batch_img.shape)
        print(batch_tgt.shape)
        break




