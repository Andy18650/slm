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

Data sources:

- `shakespeare` downloads Tiny Shakespeare from Andrej Karpathy's char-rnn repository unless `data/raw/shakespeare.txt` already exists.
- `tinystories` downloads `roneneldan/TinyStories` through Hugging Face `datasets` unless `data/raw/tinystories.txt` exists.
- `wikitext2` downloads `Salesforce/wikitext`, config `wikitext-2-raw-v1`, through Hugging Face `datasets` unless `data/raw/wikitext2.txt` exists.

Local text files always take precedence. Put custom files at `data/raw/<dataset>.txt` to use your own copy.

For laptop-friendly TinyStories experiments, start with a subset:

```bash
uv run python -m slm.prepare_data --dataset tinystories --max-chars 5000000
```

For local single-file datasets, `--train-ratio` and `--val-ratio` control contiguous train/validation/test splitting:

```bash
uv run python -m slm.prepare_data --dataset shakespeare --train-ratio 0.9 --val-ratio 0.05
```

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
