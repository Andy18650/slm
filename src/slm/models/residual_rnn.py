import torch
from torch import nn

from slm.models.custom_rnn import CustomRNNLayer


class ResidualRNNBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.rnn = CustomRNNLayer(input_dim, hidden_dim)
        self.projection = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.rnn(inputs) + self.projection(inputs)


class ResidualRNNLanguageModel(nn.Module):
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
            ResidualRNNBlock(embedding_dim if layer == 0 else hidden_dim, hidden_dim)
            for layer in range(num_layers)
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output(hidden)
