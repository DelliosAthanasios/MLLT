from tokenizer import CodeTokenizer
from dataset import CodeDataset
from models import TransformerTranslator
from metrics import evaluate_bleu, exact_match, levenshtein_distance
from trainer import ModelTrainer
from inference import ModelInference
import torch
import json
import pandas as pd
import sqlite3

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
    def save_training_data_csv(self, filepath):
        """Save training data to a CSV file."""
        df = pd.DataFrame(self.training_data, columns=['python', 'c'])
        df.to_csv(filepath, index=False)

    def load_training_data_csv(self, filepath):
        """Load training data from a CSV file."""
        try:
            df = pd.read_csv(filepath)
            self.training_data = list(df[['python', 'c']].itertuples(index=False, name=None))
            return True
        except Exception:
            return False

    def save_training_data_sqlite(self, db_path):
        """Save training data to an SQLite database."""
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS code_pairs (python TEXT, c TEXT)')
        c.execute('DELETE FROM code_pairs')
        c.executemany('INSERT INTO code_pairs (python, c) VALUES (?, ?)', self.training_data)
        conn.commit()
        conn.close()

    def load_training_data_sqlite(self, db_path):
        """Load training data from an SQLite database."""
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('SELECT python, c FROM code_pairs')
            self.training_data = c.fetchall()
            conn.close()
            return True
        except Exception:
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
        # Ensure we only save a CodeTokenizer instance
        if not isinstance(self.tokenizer, CodeTokenizer):
            raise ValueError("Tokenizer is not a CodeTokenizer instance and cannot be saved.")
        # Save tokenizer state as a dict, not as a pickled object
        tokenizer_state = {
            'token_to_id': self.tokenizer.token_to_id,
            'id_to_token': self.tokenizer.id_to_token,
            'vocab_size': self.tokenizer.vocab_size
        }
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_state': tokenizer_state,
            'model_params': {
                'embedding_dim': self.embedding_dim,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, filepath)
    def load_model(self, filepath):
        import tokenizer  # Ensure tokenizer is imported for safe_globals
        checkpoint = torch.load(filepath, map_location='cpu')
        tokenizer_state = checkpoint['tokenizer_state']
        self.tokenizer = tokenizer.CodeTokenizer()
        self.tokenizer.token_to_id = tokenizer_state['token_to_id']
        self.tokenizer.id_to_token = tokenizer_state['id_to_token']
        self.tokenizer.vocab_size = tokenizer_state['vocab_size']
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