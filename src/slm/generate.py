import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from slm.data import decode_tokens, encode_text
from slm.models import build_model
from slm.utils import select_device


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: list[int],
    itos: dict[int, str],
    max_new_chars: int,
    sequence_length: int,
    temperature: float,
    device: torch.device,
) -> str:
    model.eval()
    ids = list(input_ids)
    for _ in range(max_new_chars):
        context = torch.tensor([ids[-sequence_length:]], dtype=torch.long, device=device)
        logits = model(context)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
    return decode_tokens(ids, itos)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="To be or not to")
    parser.add_argument("--max-new-chars", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    training = config["training"]
    device = select_device(args.device)

    model_config = dict(config["model"])
    if model_config["type"].lower() == "transformer":
        model_config.setdefault("max_sequence_length", training["sequence_length"])

    model = build_model(model_config, checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    stoi = checkpoint["stoi"]
    itos = {int(key): value for key, value in checkpoint["itos"].items()}
    unknown_chars = sorted(set(args.prompt) - set(stoi))
    if unknown_chars:
        raise ValueError(f"Prompt contains characters not in the training vocabulary: {unknown_chars}")

    input_ids = encode_text(args.prompt, stoi)
    text = generate(
        model,
        input_ids,
        itos,
        args.max_new_chars,
        training["sequence_length"],
        args.temperature,
        device,
    )
    print(text)


if __name__ == "__main__":
    main()
