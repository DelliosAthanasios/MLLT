import torch
import torch.nn as nn

class TransformerTranslator(nn.Module):
    """Transformer-based sequence-to-sequence model for Python to C translation"""
    def __init__(self, vocab_size, embedding_dim=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1, max_length=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_length, embedding_dim))
        self.pos_decoder = nn.Parameter(torch.zeros(1, max_length, embedding_dim))
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src, trg):
        src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
        trg_emb = self.embedding(trg) + self.pos_decoder[:, :trg.size(1), :]
        src_key_padding_mask = (src == 0)
        tgt_key_padding_mask = (trg == 0)
        tgt_mask = self.transformer.generate_square_subsequent_mask(trg.size(1)).to(trg.device)
        out = self.transformer(
            src_emb, trg_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask
        )
        out = self.fc_out(out)
        return out

    def translate(self, src, tokenizer, max_length=512):
        self.eval()
        with torch.no_grad():
            src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1), :]
            src_key_padding_mask = (src == 0)
            generated = [tokenizer.token_to_id['<SOS>']]
            for _ in range(max_length):
                trg = torch.tensor([generated], dtype=torch.long, device=src.device)
                trg_emb = self.embedding(trg) + self.pos_decoder[:, :trg.size(1), :]
                tgt_mask = self.transformer.generate_square_subsequent_mask(trg.size(1)).to(src.device)
                out = self.transformer(
                    src_emb, trg_emb,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_mask=tgt_mask
                )
                out = self.fc_out(out)
                next_token = out[0, -1].argmax().item()
                if next_token == tokenizer.token_to_id['<EOS>']:
                    break
                generated.append(next_token)
            return generated[1:]  # skip <SOS> 