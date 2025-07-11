from tokenizer import CodeTokenizer
from dataset import CodeDataset
from models import TransformerTranslator
from metrics import evaluate_bleu, exact_match, levenshtein_distance
from trainer import ModelTrainer
from inference import ModelInference
import torch
import json

class CodeTranslator:
    """Main class for Python to C code translation"""
    def __init__(self, embedding_dim=256, hidden_size=512, num_layers=2, dropout=0.1):
        self.tokenizer = CodeTokenizer()
        self.model = None
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.training_data = []
        self.max_length = 512
        self.trainer = None
        self.inference = None
    def add_training_example(self, python_code, c_code):
        self.training_data.append((python_code, c_code))
    def clear_training_data(self):
        self.training_data = []
    def save_training_data(self, filepath):
        data = [{"python": py, "c": c} for py, c in self.training_data]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    def load_training_data(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.training_data = []
            for item in data:
                if 'python' in item and 'c' in item:
                    self.training_data.append((item['python'], item['c']))
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
    def train(self, epochs=50, batch_size=4, learning_rate=0.001, progress_callback=None):
        if not self.training_data:
            raise ValueError("No training data available. Add training examples first.")
        self.tokenizer.build_vocab(self.training_data)
        self.model = TransformerTranslator(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.embedding_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            max_length=self.max_length
        )
        dataset = CodeDataset(self.training_data, self.tokenizer)
        self.trainer = ModelTrainer(self.model, dataset, self.tokenizer, batch_size=batch_size, learning_rate=learning_rate, max_length=self.max_length)
        self.trainer.train(epochs=epochs, progress_callback=progress_callback)
        self.inference = ModelInference(self.model, self.tokenizer, max_length=self.max_length)
    def translate(self, python_code):
        if self.inference is None:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            self.inference = ModelInference(self.model, self.tokenizer, max_length=self.max_length)
        return self.inference.translate(python_code)
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'model_params': {
                'embedding_dim': self.embedding_dim,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, filepath)
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location='cpu')
        self.tokenizer = checkpoint['tokenizer']
        params = checkpoint['model_params']
        self.model = TransformerTranslator(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=params['embedding_dim'],
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=params['hidden_size'],
            dropout=params['dropout'],
            max_length=self.max_length
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.inference = ModelInference(self.model, self.tokenizer, max_length=self.max_length) 