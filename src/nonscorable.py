import json
from pathlib import Path
from transformers import BertModel, AutoTokenizer, BatchEncoding
import torch.nn as nn
import torch
import numpy as np

from src.utils import realign_embeddings, get_offsets


class NonscorableModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_dir)
        self.classifier = torch.load(Path(model_dir, "classification_layer.pt"))

    def forward(self, tokenizer_outputs: BatchEncoding, offsets: list[list[int]]):
        model_outputs = self.encoder(**tokenizer_outputs)
        logits = realign_embeddings(model_outputs, offsets)
        logits = logits[:, 0, :]
        logits = self.classifier(logits)

        return logits


class NonscorablePipeline:
    def __init__(self, model_dir: str):
        self.model = NonscorableModel(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        with open(Path(model_dir, "labels.json")) as f:
            self.labels = json.load(f)

    def get_prediction(self, text: str) -> dict:
        tokenizer_output = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = get_offsets(text, tokenizer_output)

        logits = self.model(tokenizer_output, offsets)

        probs = nn.functional.softmax(logits, dim=1).detach().numpy()[0]
        label = self.labels[str(np.argmax(probs))]

        return {
            "logits": logits.detach().numpy()[0].tolist(),
            "probs": probs.tolist(),
            "predicted_label": label,
        }
