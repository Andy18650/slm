import torch
from torch import nn
from torch.nn import init


class SplitStateRNNLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight_ih = nn.Parameter(torch.empty(input_dim, hidden_dim * 2))
        self.weight_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim * 2))
        self.bias_ih = nn.Parameter(torch.empty(hidden_dim * 2))
        self.bias_hh = nn.Parameter(torch.empty(hidden_dim * 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.hidden_dim**-0.5
        init.uniform_(self.weight_ih, -bound, bound)
        init.uniform_(self.weight_hh, -bound, bound)
        init.uniform_(self.bias_ih, -bound, bound)
        init.uniform_(self.bias_hh, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape
        state = inputs.new_zeros(batch_size, self.hidden_dim)
        outputs = []
        for time_step in range(sequence_length):
            combined = torch.tanh(
                inputs[:, time_step] @ self.weight_ih
                + self.bias_ih
                + state @ self.weight_hh
                + self.bias_hh
            )
            # First half feeds the next layer; second half is carried to the next time step.
            output, state = combined.chunk(2, dim=-1)
            outputs.append(output)
        return torch.stack(outputs, dim=1)


class SplitStateRNNLanguageModel(nn.Module):
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
            SplitStateRNNLayer(embedding_dim if layer == 0 else hidden_dim, hidden_dim)
            for layer in range(num_layers)
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output(hidden)
