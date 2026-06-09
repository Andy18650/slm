# Small Language Model Architecture Comparison

This project trains small token-level language models to compare vanilla RNN, LSTM, GRU, and Transformer architectures.

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
uv run python -m slm.prepare_data --dataset shakespeare --tokenizer char
```

For token-level experiments, prepare BPE or word-token data:

```bash
uv run python -m slm.prepare_data --dataset shakespeare --tokenizer bpe --vocab-size 8000
uv run python -m slm.prepare_data --dataset shakespeare --tokenizer word --lowercase
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
uv run python -m slm.prepare_data --dataset tinystories --tokenizer bpe --max-chars 5000000 --vocab-size 8000
```

Tokenizer choices:

- `char` preserves the original character-level experiment.
- `word` splits into words, punctuation, and whitespace tokens.
- `bpe` trains a byte-level BPE tokenizer on the training split.

Optional lowercasing can reduce vocabulary size:

```bash
uv run python -m slm.prepare_data --dataset shakespeare --tokenizer word --lowercase
```

For local single-file datasets, `--train-ratio` and `--val-ratio` control contiguous train/validation/test splitting:

```bash
uv run python -m slm.prepare_data --dataset shakespeare --train-ratio 0.9 --val-ratio 0.05
```

## Train

```bash
uv run python -m slm.train --config configs/rnn.yaml --dataset shakespeare --tokenizer bpe --wandb-project slm-architecture-comparison
uv run python -m slm.train --config configs/lstm.yaml --dataset shakespeare --tokenizer bpe --wandb-project slm-architecture-comparison
uv run python -m slm.train --config configs/gru.yaml --dataset shakespeare --tokenizer bpe --wandb-project slm-architecture-comparison
uv run python -m slm.train --config configs/transformer.yaml --dataset shakespeare --tokenizer bpe --wandb-project slm-architecture-comparison
```

The config file contains only model and training hyperparameters. Dataset choice, tokenizer choice, W&B grouping, and output locations are runtime parameters.

By default, training reads `data/processed/<dataset>_<tokenizer>.pt` and writes to `runs/<dataset>/<tokenizer>/<model-signature>/`.

Disable W&B for a run:

```bash
uv run python -m slm.train --config configs/lstm.yaml --dataset shakespeare --tokenizer bpe --wandb-project slm-architecture-comparison --no-wandb
```

Use a different W&B project to group a new set of runs:

```bash
uv run python -m slm.train --config configs/lstm.yaml --dataset tinystories --tokenizer bpe --wandb-project slm-tinystories-subset
```

## Generate

```bash
uv run python -m slm.generate --checkpoint runs/shakespeare/bpe/lstm_embedding_dim-256_hidden_dim-256_num_layers-2/best.pt --prompt "To be or not to" --max-new-tokens 200
```

## Outputs

Each run writes checkpoints, metrics, and samples under `runs/<experiment_name>/`.
