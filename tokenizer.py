import re
from collections import Counter

class CodeTokenizer:
    """Tokenizer for Python and C code"""
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
    def build_vocab(self, code_pairs):
        all_tokens = []
        for py_code, c_code in code_pairs:
            py_tokens = self.tokenize(py_code)
            c_tokens = self.tokenize(c_code)
            all_tokens.extend(py_tokens + c_tokens)
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        token_counts = Counter(all_tokens)
        vocab = special_tokens + [token for token, count in token_counts.most_common(10000)]
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}
        self.vocab_size = len(vocab)
    def tokenize(self, code):
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return tokens
    def encode(self, code):
        tokens = self.tokenize(code)
        return [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]
    def decode(self, token_ids):
        tokens = [self.id_to_token.get(id, '<UNK>') for id in token_ids]
        return ' '.join(tokens) 