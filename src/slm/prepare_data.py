import argparse
from pathlib import Path

import requests
import torch


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def read_or_download_dataset(dataset: str, raw_dir: Path) -> str:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{dataset}.txt"

    if path.exists():
        return path.read_text(encoding="utf-8")

    if dataset == "shakespeare":
        response = requests.get(SHAKESPEARE_URL, timeout=30)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
        return response.text

    raise FileNotFoundError(
        f"Expected local dataset at {path}. Only shakespeare is downloaded automatically."
    )


def prepare_character_data(text: str, dataset: str, output_path: Path) -> None:
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)

    train_end = int(0.9 * len(encoded))
    val_end = int(0.95 * len(encoded))
    payload = {
        "dataset": dataset,
        "level": "char",
        "vocab_size": len(chars),
        "stoi": stoi,
        "itos": itos,
        "train": encoded[:train_end],
        "val": encoded[train_end:val_end],
        "test": encoded[val_end:],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved {dataset} character data to {output_path}")
    print(f"Characters: {len(encoded):,}; vocabulary: {len(chars)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare character-level language data.")
    parser.add_argument(
        "--dataset",
        choices=["shakespeare", "tinystories", "wikitext2"],
        default="shakespeare",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = read_or_download_dataset(args.dataset, Path(args.raw_dir))
    output_path = Path(args.output_dir) / f"{args.dataset}_char.pt"
    prepare_character_data(text, args.dataset, output_path)


if __name__ == "__main__":
    main()
