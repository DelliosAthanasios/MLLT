import torch

class ModelInference:
    """Handles model inference/translation logic."""
    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def translate(self, python_code):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        tokens = [self.tokenizer.token_to_id['<SOS>']] + self.tokenizer.encode(python_code) + [self.tokenizer.token_to_id['<EOS>']]
        tokens = tokens[:self.max_length]
        tokens = tokens + [self.tokenizer.token_to_id['<PAD>']] * (self.max_length - len(tokens))
        src_tensor = torch.tensor([tokens], dtype=torch.long)
        result_tokens = self.model.translate(src_tensor, self.tokenizer)
        return self.tokenizer.decode(result_tokens) 