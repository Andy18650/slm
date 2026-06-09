import torch
from torch import nn
from torch.nn import init


class CustomRNNLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight_ih = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.weight_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.hidden_dim**-0.5
        init.uniform_(self.weight_ih, -bound, bound)
        init.uniform_(self.weight_hh, -bound, bound)
        init.uniform_(self.bias_ih, -bound, bound)
        init.uniform_(self.bias_hh, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        hidden = inputs.new_zeros(batch_size, self.hidden_dim)
        outputs = []
        for time_step in range(sequence_length):
            # Vanilla RNN recurrence: h_t = tanh(x_t W_ih + b_ih + h_{t-1} W_hh + b_hh).
            hidden = torch.tanh(
                inputs[:, time_step] @ self.weight_ih
                + self.bias_ih
                + hidden @ self.weight_hh
                + self.bias_hh
            )
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class CustomRNNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList(
            CustomRNNLayer(embedding_dim if layer == 0 else hidden_dim, hidden_dim)
            for layer in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        for layer_index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if layer_index < len(self.layers) - 1:
                hidden = self.dropout(hidden)
        return self.output(self.dropout(hidden))
