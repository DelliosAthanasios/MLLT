import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ModelTrainer:
    """Handles model training loop and optimization."""
    def __init__(self, model, dataset, tokenizer, batch_size=4, learning_rate=0.001, max_length=512, device=None):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id['<PAD>'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, epochs=50, progress_callback=None):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (src, trg) in enumerate(self.dataloader):
                if self.device is not None:
                    src = src.to(self.device)
                    trg = trg.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(src, trg[:, :-1])
                output = output.reshape(-1, self.tokenizer.vocab_size)
                target = trg[:, 1:].reshape(-1)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.dataloader)
            if progress_callback:
                progress_callback(epoch + 1, epochs, avg_loss) 