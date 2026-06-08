# Small Language Model Architecture Comparison

This project trains small character-level language models to compare vanilla RNN, LSTM, GRU, and Transformer architectures.

The trainer is shared across all models. For recurrent models, each fixed-length batch is a truncated BPTT window. For the Transformer, the same batch is a causal context block. This keeps the optimizer, data pipeline, metrics, logging, and checkpointing identical across architectures.

## Setup

```bash
uv sync
```

Optional W&B setup:

```bash
uv run wandb login
```

## Prepare Data

```bash
uv run python -m slm.prepare_data --dataset shakespeare
```

Supported datasets:

- `shakespeare`
- `tinystories`
- `wikitext2`

TinyStories and WikiText-2 are prepared from local text files by default. Put them in `data/raw/tinystories.txt` or `data/raw/wikitext2.txt`.

## Train

```bash
uv run python -m slm.train --config configs/rnn_shakespeare.yaml
uv run python -m slm.train --config configs/lstm_shakespeare.yaml
uv run python -m slm.train --config configs/gru_shakespeare.yaml
uv run python -m slm.train --config configs/transformer_shakespeare.yaml
```

Disable W&B for a run:

```bash
uv run python -m slm.train --config configs/lstm_shakespeare.yaml --no-wandb
```

## Generate

```bash
uv run python -m slm.generate --checkpoint runs/lstm_shakespeare/best.pt --prompt "To be or not to" --max-new-chars 500
```

## Outputs

Each run writes checkpoints, metrics, and samples under `runs/<experiment_name>/`.
