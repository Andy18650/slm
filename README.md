# Small Language Model Architecture Comparison

This project trains small BPE-tokenized language models to compare vanilla RNN, LSTM, GRU, and Transformer architectures.

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
uv run python -m slm.prepare_data --dataset shakespeare --vocab-size 8000
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
uv run python -m slm.prepare_data --dataset tinystories --max-chars 5000000 --vocab-size 8000
```

TinyStories and WikiText-2 require `--max-chars` by default because full Hugging Face datasets can require enough memory to destabilize WSL. To intentionally process a full Hugging Face dataset, add `--allow-full-dataset`.

Optional lowercasing can reduce vocabulary size:

```bash
uv run python -m slm.prepare_data --dataset shakespeare --lowercase --vocab-size 8000
```

For local single-file datasets, `--train-ratio` and `--val-ratio` control contiguous train/validation/test splitting:

```bash
uv run python -m slm.prepare_data --dataset shakespeare --train-ratio 0.9 --val-ratio 0.05
```

## Train

```bash
uv run python -m slm.train --config configs/rnn.yaml --dataset shakespeare --wandb-project slm-architecture-comparison
uv run python -m slm.train --config configs/custom_rnn.yaml --dataset shakespeare --wandb-project slm-architecture-comparison
uv run python -m slm.train --config configs/lstm.yaml --dataset shakespeare --wandb-project slm-architecture-comparison
uv run python -m slm.train --config configs/gru.yaml --dataset shakespeare --wandb-project slm-architecture-comparison
uv run python -m slm.train --config configs/transformer.yaml --dataset shakespeare --wandb-project slm-architecture-comparison
```

The config file contains only model and training hyperparameters. Dataset choice, W&B grouping, and output locations are runtime parameters.

By default, training reads `data/processed/<dataset>_bpe.pt` and writes to `runs/<dataset>/<model-signature>/`.

Disable W&B for a run:

```bash
uv run python -m slm.train --config configs/lstm.yaml --dataset shakespeare --wandb-project slm-architecture-comparison --no-wandb
```

Use a different W&B project to group a new set of runs:

```bash
uv run python -m slm.train --config configs/lstm.yaml --dataset tinystories --wandb-project slm-tinystories-subset
```

## Generate

```bash
uv run python -m slm.generate --checkpoint runs/shakespeare/lstm_embedding_dim-256_hidden_dim-256_num_layers-2/best.pt --prompt "To be or not to" --max-new-tokens 200
```

## Outputs

Each run writes checkpoints and metrics under `runs/<dataset>/<model-signature>/` unless `--output-dir` is provided.
