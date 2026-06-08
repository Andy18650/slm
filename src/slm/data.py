from pathlib import Path

import torch


class CharBatcher:
    def __init__(
        self,
        token_ids: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> None:
        if len(token_ids) <= sequence_length + 1:
            raise ValueError("Dataset split is too small for the configured sequence_length.")
        self.token_ids = token_ids.to(torch.long)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device

    def __iter__(self):
        max_start = len(self.token_ids) - self.sequence_length - 1
        starts = torch.randint(max_start, (self.batch_size,))
        x = torch.stack([self.token_ids[start : start + self.sequence_length] for start in starts])
        y = torch.stack(
            [self.token_ids[start + 1 : start + self.sequence_length + 1] for start in starts]
        )
        yield x.to(self.device), y.to(self.device)


def get_batch(
    token_ids: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return next(iter(CharBatcher(token_ids, batch_size, sequence_length, device)))


def load_processed_data(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def encode_text(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[char] for char in text]


def decode_tokens(token_ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[int(token_id)] for token_id in token_ids)
