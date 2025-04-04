import json
from pathlib import Path
from transformers import BertModel, AutoTokenizer, BatchEncoding
import torch.nn as nn
import torch
import numpy as np
import spacy

from src.utils import get_offsets, realign_embeddings, reshape_tensor

class IdeaDetectionModel(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_dir)
        self.pooler = torch.load(Path(model_dir, 'encoder.pt'))
        self.label_projection_layer = torch.load(Path(model_dir, 'label_projection_layer_module.pt'))
    
    def forward(self, tokenizer_outputs: BatchEncoding, offsets: list[list[int]]) -> dict:
        model_outputs = self.encoder(**tokenizer_outputs)
        logits = realign_embeddings(model_outputs, offsets)
        logits = self.pooler(logits)
        logits = self.label_projection_layer(logits[0])
        
        return logits
    
    def label_projection_layer(self, inputs: torch.Tensor):
        """
        Apply the label projection layer module and reshaping steps

        The label projection layer module in Allennlp is wrapped in a module called TimeDistributed which performs data
        reshaping before and after calling the label projection module. This method performs the same processing
        to ensure that the output is similar to that of the Allennlp model.

        Original code can be found here:
        https://github.com/allenai/allennlp/blob/main/allennlp/modules/time_distributed.py

        """
        # Need some input to then get the batch_size and time_steps.
        some_input = inputs[-1]

        reshaped_inputs = [reshape_tensor(input_tensor) for input_tensor in inputs]
        reshaped_outputs = self.label_projection_layer(*reshaped_inputs)

        # Now get the output back into the right shape.
        # (batch_size, time_steps, **output_size)
        tuple_output = True
        if not isinstance(reshaped_outputs, tuple):
            tuple_output = False
            reshaped_outputs = (reshaped_outputs,)

        outputs = []
        for reshaped_output in reshaped_outputs:
            new_size = some_input.size()[:2] + reshaped_output.size()[1:]
            outputs.append(reshaped_output.contiguous().view(new_size))

        if not tuple_output:
            outputs = outputs[0]

        return outputs


class IdeaDetectionPipeline:
    def __init__(self, model_dir: str):
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.spacy_tokenizer = spacy.load('en_core_web_sm')
        self.model = IdeaDetectionModel(model_dir)
        with open(Path(model_dir, "labels.json"), "r") as f:
            self.labels = json.load(f)
    
    def get_prediction(self, text: str) -> dict:
        # allennlp tokenizer
        text = ' '.join(text.split())
        spacy_tokenizer_output = self.spacy_tokenizer(text)
        spacy_tokens = [token.text for token in spacy_tokenizer_output]
        input_text = ' '.join(spacy_tokens)
        
        # huggingface tokenizer
        transformer_tokenizer_output = self.transformer_tokenizer(input_text, add_special_tokens=True, max_length=512, return_tensors='pt')
        offsets = get_offsets(input_text, transformer_tokenizer_output)
        
        # forward
        logits = self.model(transformer_tokenizer_output, offsets)
        
        # post processing
        label_probabilities = torch.sigmoid(logits).tolist()
        predicted_labels = self.map_logits_to_labels(logits)
        
        return {
            'logits': logits.tolist(),
            'label_probabilities': label_probabilities,
            'words': spacy_tokens,
            'predicted_labels': predicted_labels
        }
    
    def map_logits_to_labels(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        label_probabilities = torch.sigmoid(logits).cpu().data.numpy()
        argmax_indices = np.argwhere(label_probabilities > 0.5)
        predicted_labels = [[] for word in range(label_probabilities.shape[0])]
        
        for word_index, label_index in argmax_indices:
            label = self.labels[str(label_index)]
            predicted_labels[word_index].append(label)
        
        return predicted_labels
        
        