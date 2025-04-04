from pathlib import Path
from transformers import BertModel, AutoTokenizer, BatchEncoding
import torch.nn as nn
import torch


class MeanPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.mean(hidden_states, dim=1)


class ScoringModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        
        self.encoder = BertModel.from_pretrained(model_dir)
        self.feedforward = torch.load(Path(model_dir, 'feedforward.pt'))
        self.ff_activation = torch.load(Path(model_dir, 'ff_activation.pt'))
        pooler_path = Path(model_dir, 'pooler.pt')
        if pooler_path.exists():
            self.pooler = torch.load(pooler_path)
        else:
            self.pooler = MeanPooling()
    
    def forward(self, tokenizer_outputs: BatchEncoding) -> torch.Tensor:
        model_outputs = self.encoder(**tokenizer_outputs)
        logits = self.pooler(model_outputs.last_hidden_state)
        logits = self.feedforward(logits)
        logits = self.ff_activation(logits)
    
        return logits


class ScoringPipeline:
    def __init__(self, model_dir: str):
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = ScoringModel(model_dir)
    
    @staticmethod
    def unscale(logits: torch.Tensor, min_score: int, max_score: int) -> torch.Tensor:
        unscaled = logits * (max_score - min_score) + min_score
        unscaled = torch.as_tensor(unscaled, device=logits.device)
        
        return unscaled
        
    def get_prediction(self, text: str, min_score: int, max_score: int) -> dict:
        inputs = self.transformer_tokenizer(text, return_tensors='pt')
        logits = self.model(inputs)
        logits_value = float(logits[0])
        
        # post processing
        logits_unscaled = ScoringPipeline.unscale(logits, min_score, max_score)
        logits_unscaled_value = float(logits_unscaled[0])
        formatted_predicted_score = format(logits_unscaled_value, '.4f')
        
        return {
            'logit': logits_value,
            'logit_unscaled': logits_unscaled_value,
            "formatted_predicted_score": formatted_predicted_score
        }