import pandas as pd
from collections import Counter
from typing import Optional, List
from rdkit import Chem
import pickle as pkl
import re


class Vocabulary:

    def __init__(self, token_list: Optional[List[str]] = None,
                 pad_token='<pad>', bos_token='<bos>',
                 eos_token='<eos>', unk_token='<unk>'):

        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        self._token_to_idx = {}
        self._idx_to_token = {}
        self._current_idx = 0

        for token in [pad_token, bos_token, eos_token, unk_token]:
            if token is not None:
                self.add(token)

        if token_list:
            for token in token_list:
                self.add(token)

        self.unk_index = self[unk_token] if unk_token else None
        self.pad_index = self[pad_token]
        self.eos_index = self[eos_token]
        self.bos_index = self[bos_token]

    def __getitem__(self, key):
        """支持 token->idx 和 idx->token 双向访问。"""
        if isinstance(key, str):
            return self._token_to_idx[key]
        else:
            return self._idx_to_token[key]

    def __len__(self):
        return len(self._token_to_idx)

    def __contains__(self, item):
        """支持 in 操作符，对 token 字符串有效，对索引也有效"""
        if isinstance(item, str):
            return item in self._token_to_idx
        else:
            return item in self._idx_to_token

    def add(self, token):
        if token in self._token_to_idx.keys():
            return self._token_to_idx[token]
        idx = self._current_idx
        self._token_to_idx[token] = idx
        self._idx_to_token[idx] = token
        self._current_idx += 1
        return idx

    def encode(self, tokens):
        "token -> 索引"
        return [self._token_to_idx.get(t, self.unk_index) for t in tokens]

    def decode(self, indices):
        return [self._idx_to_token.get(i, self.unk_token) for i in indices]

    def word2idx(self):
        return self._token_to_idx.copy()

    def get_dictionary(self):
        return {
            "tokens": self.word2idx(),
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
        }

    @classmethod
    def load_from_dictionary(cls, dictionary):
        vocab = cls(
            token_list=None,
            pad_token=dictionary['pad_token'],
            bos_token=dictionary['bos_token'],
            eos_token=dictionary['eos_token'],
            unk_token=dictionary['unk_token']
        )
        vocab._token_to_idx.clear()
        vocab._idx_to_token.clear()
        vocab._current_idx = 0
        for token, idx in sorted(dictionary['tokens'].items(), key=lambda x: x[1]):
            vocab._token_to_idx[token] = idx
            vocab._idx_to_token[idx] = token
            vocab._current_idx = max(vocab._current_idx, idx + 1)

        if vocab.unk_token:
            vocab.unk_index = vocab[vocab.unk_token]
        return vocab

SMILES_PATTERN = re.compile(
r'(\[[^\[\]]+]|'
r'Br|Cl|Si|Li|Na|Mg|Al|Ca|Sc|Ti|Cr|Mn|Fe|Co|He|'
r'Ni|Cu|Zn|Ga|Ge|As|Se|'
r'C|H|N|O|S|P|F|I|B|'
r'b|c|n|o|s|p|f|i|'
r'\(|\)|\.|'
r'=|#|-|\+|\\|/|:|~|'
r'@@?|\?|'
r'[0-9]+|%[0-9]{2}|'
r'[A-Z][a-z]?|[a-z])'
)

class Tokenizer:
    def __init__(self, pattern, vocab: Vocabulary):
        self.PATTERN = pattern
        self.VOCAB = vocab

    def batch_tokenize(self, smiles_list, pad: bool=True):
        encoded_tokens = []
        len_max = 0
        for smiles in smiles_list:
            tokens = self.PATTERN.findall(smiles)
            encoded = [self.VOCAB.bos_index] + self.VOCAB.encode(tokens) + [self.VOCAB.eos_index]
            encoded_tokens.append(encoded)
            len_max = max(len_max, len(encoded))

        if pad:
            for seq in encoded_tokens:
                len_seq = len(seq)
                if len_seq < len_max:
                    seq.extend([self.VOCAB.pad_index] * (len_max - len_seq))

        return encoded_tokens

    def tokenize(self, smiles):
        return self.VOCAB.encode(self.PATTERN.findall(smiles))


if __name__ == "__main__":

    csv_path = "mol_img/train_labels.csv"
    #数据列：image_id, InChI
    chunk_size = 10000

    all_tokens = []
    failed_inchi = {}
    train_smiles = {
        'image_id': [],
        'smiles': []
    }

    chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size)
    for idx, chunk in enumerate(chunk_iter):

        inchi_list = chunk["InChI"]
        id_list = list(chunk["image_id"])

        smiles_list = []
        valid_ids = []

        for inchi, img_id in zip(inchi_list, id_list):
            try:
                mol = Chem.MolFromInchi(inchi)
                smiles = Chem.MolToSmiles(mol)

                smiles_list.append(smiles)
                valid_ids.append(img_id)

                all_tokens.extend(SMILES_PATTERN.findall(smiles))
            except Exception as e:
                print(f"tokenize inchi {inchi} failed - {e}")
                failed_inchi[img_id] = inchi

        train_smiles['image_id'].extend(valid_ids)
        train_smiles['smiles'].extend(smiles_list)
        print(f"chunk {idx} finished")

    train_df = pd.DataFrame(train_smiles)
    train_df.to_csv("mol_img/train_smiles.csv", index=False)

    if len(failed_inchi) > 0:
        print("failed inchi:")
        for id, inchi in failed_inchi.items():
            print(f"id: {id} | inchi: {inchi}")
        with open("failed_inchi.pkl", "wb") as f:
            pkl.dump(failed_inchi, f)


    counter = Counter(all_tokens)
    token_list = [token for token, cnt in counter.items() if cnt >= 2]

    vocab = Vocabulary(token_list)

    vocab_dict = vocab.get_dictionary()
    with open("vocabulary_dictionary.pkl", "wb") as f:
        pkl.dump(vocab_dict, f)

    with open("vocabulary_dictionary.pkl", "rb") as f:
        loaded_dict = pkl.load(f)
    vocab = Vocabulary.load_from_dictionary(loaded_dict)
    print(vocab.word2idx())


