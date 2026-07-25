import argparse
from datetime import datetime
from pathlib import Path

import torch

from devices import DEVICE_CHOICES, choose_device, describe_device
from model import TinyTransformerLanguageModel
from tokenization import tokenizer_from_checkpoint
from train import DEFAULT_DATA_PATH, load_training_text


ROOT = Path(__file__).parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "tiny_transformer_best.pt"
RUNS_DIR = ROOT / "runs"
DEFAULT_PROMPTS = (
    "My dear Watson",
    "Holmes looked",
    "The inspector",
    "It was a curious case",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved tiny LLM checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Checkpoint to evaluate. Default: {CHECKPOINT_PATH}",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Cleaned training file or folder. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help="Evaluation device. Default: auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for loss evaluation. Default: 32",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=50,
        help="Number of deterministic random batches per split. Default: 50",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed for deterministic evaluation batches and samples. Default: 1337",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=240,
        help="New tokens to generate for each prompt. Default: 240",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for report samples. Default: 0.8",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling filter. Use 0 to disable. Default: 20",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling filter. Use 1.0 to disable. Default: 0.95",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Prompt to include in the report. Can be passed multiple times.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save a Markdown report under runs/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Specific report path. Implies --save.",
    )
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def split_data(tokens: list[int], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(tokens, dtype=torch.long, device=device)
    split_idx = int(0.9 * len(data))
    return data[:split_idx], data[split_idx:]


def get_eval_batch(
    data: torch.Tensor,
    context_length: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        len(data) - context_length,
        (batch_size, 1),
        generator=generator,
        device="cpu",
    ).to(data.device)
    offsets = torch.arange(context_length + 1, device=data.device)
    windows = data[starts + offsets]
    return windows[:, :-1], windows[:, 1:]


@torch.inference_mode()
def estimate_split_loss(
    model: TinyTransformerLanguageModel,
    data: torch.Tensor,
    context_length: int,
    batch_size: int,
    eval_batches: int,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses = torch.zeros(eval_batches)
    for step in range(eval_batches):
        x, y = get_eval_batch(data, context_length, batch_size, generator)
        _, loss = model(x, y)
        losses[step] = loss.item()
    return losses.mean().item()


@torch.inference_mode()
def generate_sample(
    model: TinyTransformerLanguageModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    device: str,
) -> str:
    start_tokens = tokenizer.encode(prompt) or tokenizer.encode("\n") or [0]
    start = torch.tensor([start_tokens], dtype=torch.long, device=device)
    generated = model.generate(
        start,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )[0].tolist()
    return tokenizer.decode(generated)


def format_report(
    checkpoint_path: Path,
    data_path: Path,
    data_files: list[Path],
    checkpoint: dict,
    tokenizer_name: str,
    device_description: str,
    token_count: int,
    train_loss: float,
    val_loss: float,
    parameter_count: int,
    prompts_to_samples: list[tuple[str, str]],
    args,
) -> str:
    lines = [
        "# Tiny LLM Evaluation",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Data: `{data_path}` ({len(data_files)} file(s))",
        f"- Device: {device_description}",
        f"- Tokenizer: `{tokenizer_name}`",
        f"- Checkpoint step: `{checkpoint.get('iteration', 'unknown')}`",
        f"- Checkpoint val loss: `{checkpoint.get('val_loss', 'unknown')}`",
        f"- Measured train loss: `{train_loss:.4f}`",
        f"- Measured val loss: `{val_loss:.4f}`",
        f"- Tokens in corpus: `{token_count:,}`",
        f"- Vocab size: `{checkpoint['vocab_size']:,}`",
        f"- Context length: `{checkpoint['context_length']}`",
        f"- Embedding dim: `{checkpoint['embedding_dim']}`",
        f"- Heads: `{checkpoint['num_heads']}`",
        f"- Layers: `{checkpoint.get('num_layers', 1)}`",
        f"- Dropout: `{checkpoint.get('dropout', 0.0)}`",
        f"- Activation: `{checkpoint.get('activation', 'relu')}`",
        f"- Weight tying: `{checkpoint.get('tie_weights', False)}`",
        f"- Gradient clip: `{checkpoint.get('grad_clip', 'unknown')}`",
        f"- Parameters: `{parameter_count:,}`",
        f"- Eval batches: `{args.eval_batches}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Seed: `{args.seed}`",
        f"- Sampling: temperature `{args.temperature}`, top-k `{args.top_k}`, top-p `{args.top_p}`",
        "",
        "## Samples",
        "",
    ]

    for prompt, sample in prompts_to_samples:
        lines.extend(
            [
                f"### `{prompt}`",
                "",
                "```text",
                sample.strip(),
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def save_report(report: str, output_path: Path | None) -> Path:
    if output_path is None:
        RUNS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RUNS_DIR / f"eval_{timestamp}.md"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(report, encoding="utf-8")
    return output_path


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be greater than 0")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    text, data_files = load_training_text(args.data)
    tokens = tokenizer.encode(text)
    train_data, val_data = split_data(tokens, device)

    model = TinyTransformerLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embedding_dim=checkpoint["embedding_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint.get("num_layers", 1),
        dropout=checkpoint.get("dropout", 0.0),
        activation=checkpoint.get("activation", "relu"),
        tie_weights=checkpoint.get("tie_weights", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    train_loss = estimate_split_loss(
        model,
        train_data,
        checkpoint["context_length"],
        args.batch_size,
        args.eval_batches,
        generator,
    )
    val_loss = estimate_split_loss(
        model,
        val_data,
        checkpoint["context_length"],
        args.batch_size,
        args.eval_batches,
        generator,
    )

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p < 1.0 else None
    prompts = args.prompts or list(DEFAULT_PROMPTS)
    samples = [
        (
            prompt,
            generate_sample(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
                top_k,
                top_p,
                device,
            ),
        )
        for prompt in prompts
    ]

    report = format_report(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        data_files=data_files,
        checkpoint=checkpoint,
        tokenizer_name=tokenizer.name,
        device_description=describe_device(device),
        token_count=len(tokens),
        train_loss=train_loss,
        val_loss=val_loss,
        parameter_count=count_parameters(model),
        prompts_to_samples=samples,
        args=args,
    )
    print(report)

    if args.save or args.output:
        output_path = save_report(report, args.output)
        print(f"\nSaved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
