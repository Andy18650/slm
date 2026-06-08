import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

from slm.data import get_batch, load_processed_data
from slm.models import build_model
from slm.utils import count_parameters, load_yaml, perplexity, save_json, select_device, set_seed


def maybe_init_wandb(config: dict, enabled: bool):
    wandb_config = config.get("wandb", {})
    if not enabled or not wandb_config.get("enabled", False):
        return None

    import wandb

    return wandb.init(
        project=wandb_config.get("project", "slm-architecture-comparison"),
        name=wandb_config.get("name", config.get("name")),
        config=config,
        mode=wandb_config.get("mode", "online"),
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    eval_iters: int,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(token_ids, batch_size, sequence_length, device, generator=generator)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def write_metric_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict,
    data_meta: dict,
    step: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stoi": data_meta["stoi"],
            "itos": data_meta["itos"],
            "vocab_size": data_meta["vocab_size"],
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def train(config: dict, disable_wandb: bool = False) -> None:
    training = config["training"]
    set_seed(training.get("seed", 42))
    device = select_device(training.get("device", "auto"))
    data = load_processed_data(config["data_path"])

    model_config = dict(config["model"])
    if model_config["type"].lower() == "transformer":
        model_config.setdefault("max_sequence_length", training["sequence_length"])

    model = build_model(model_config, vocab_size=data["vocab_size"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training.get("weight_decay", 0.0),
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")

    run = maybe_init_wandb(config, enabled=not disable_wandb)
    param_count = count_parameters(model)
    if run is not None:
        run.summary["parameters"] = param_count
        run.summary["device"] = str(device)

    steps_per_epoch = training["steps_per_epoch"]
    total_steps = training["epochs"] * steps_per_epoch
    eval_interval = training.get("eval_interval", steps_per_epoch)
    eval_iters = training.get("eval_iters", 20)
    best_val_loss = float("inf")
    start_time = time.perf_counter()

    progress = trange(1, total_steps + 1, desc=config.get("name", "train"))
    for step in progress:
        x, y = get_batch(
            data["train"],
            training["batch_size"],
            training["sequence_length"],
            device,
        )
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_clip = training.get("grad_clip")
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == total_steps:
            val_loss = evaluate(
                model,
                data["val"],
                training["batch_size"],
                training["sequence_length"],
                device,
                eval_iters,
                seed=training.get("seed", 42),
            )
            elapsed = time.perf_counter() - start_time
            row = {
                "step": step,
                "epoch": step / steps_per_epoch,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "val_perplexity": perplexity(val_loss),
                "elapsed_seconds": elapsed,
                "parameters": param_count,
            }
            write_metric_row(output_dir / "metrics.csv", row)
            progress.set_postfix(train_loss=f"{loss.item():.3f}", val_loss=f"{val_loss:.3f}")

            if run is not None:
                run.log(row, step=step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(output_dir / "best.pt", model, config, data, step, val_loss)

    save_checkpoint(output_dir / "last.pt", model, config, data, total_steps, best_val_loss)
    if run is not None:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a character-level language model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_yaml(args.config), disable_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
