import math

import torch
import torch.nn.functional as F
from torch import nn


class AttentionRNNLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output_and_value = nn.Linear(input_dim + hidden_dim, hidden_dim * 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        queries = self.query(inputs)
        keys = self.key(inputs)
        values = []
        outputs = []

        for time_step in range(sequence_length):
            if time_step == 0:
                context = inputs.new_zeros(batch_size, self.hidden_dim)
            else:
                scores = torch.bmm(
                    queries[:, time_step].unsqueeze(1),
                    keys[:, :time_step].transpose(1, 2),
                ).squeeze(1)
                weights = F.softmax(scores / math.sqrt(self.hidden_dim), dim=-1)
                context = torch.bmm(weights.unsqueeze(1), torch.stack(values, dim=1)).squeeze(1)

            combined = torch.cat((inputs[:, time_step], context), dim=-1)
            output, value = torch.tanh(self.output_and_value(combined)).chunk(2, dim=-1)
            outputs.append(output)
            values.append(value)

        return torch.stack(outputs, dim=1)


class AttentionRNNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList(
            AttentionRNNLayer(embedding_dim if layer == 0 else hidden_dim, hidden_dim)
            for layer in range(num_layers)
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output(hidden)
