# models/text_encoder.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, texts):
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self.encoder(**encoded_input)
        return mean_pooling(model_output, encoded_input["attention_mask"])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element is the token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def load_text_encoder(device):
    encoder = TextEncoder()
    encoder.to(device)
    encoder.eval()
    return encoder

def encode_text(text, encoder):
    if isinstance(text, str):
        text = [text]
    return encoder(text).squeeze(0)
