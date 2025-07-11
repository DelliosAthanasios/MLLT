import torch
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    """Dataset for Python to C code translation"""
    def __init__(self, code_pairs, tokenizer, max_length=512):
        self.code_pairs = code_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.code_pairs)
    def __getitem__(self, idx):
        py_code, c_code = self.code_pairs[idx]
        py_tokens = [self.tokenizer.token_to_id['<SOS>']] + self.tokenizer.encode(py_code) + [self.tokenizer.token_to_id['<EOS>']]
        c_tokens = [self.tokenizer.token_to_id['<SOS>']] + self.tokenizer.encode(c_code) + [self.tokenizer.token_to_id['<EOS>']]
        py_tokens = self.pad_sequence(py_tokens, self.max_length)
        c_tokens = self.pad_sequence(c_tokens, self.max_length)
        return torch.tensor(py_tokens, dtype=torch.long), torch.tensor(c_tokens, dtype=torch.long)
    def pad_sequence(self, seq, max_length):
        if len(seq) > max_length:
            return seq[:max_length]
        return seq + [self.tokenizer.token_to_id['<PAD>']] * (max_length - len(seq)) 