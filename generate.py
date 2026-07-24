import argparse
from pathlib import Path

import torch

from devices import DEVICE_CHOICES, choose_device, describe_device
from model import TinyTransformerLanguageModel
from tokenization import tokenizer_from_checkpoint


ROOT = Path(__file__).parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "tiny_transformer_best.pt"
TEMPERATURE = 0.8
TOP_K = 20
TOP_P = 0.95
MAX_NEW_TOKENS = 500


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Optional starting text to condition generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature. Lower is safer; higher is more varied.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Keep only the k most likely next tokens. Use 0 to disable.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=TOP_P,
        help="Keep the smallest set of tokens whose probability mass reaches p. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help=(
            "Inference device. 'auto' prefers CUDA, then Apple Silicon MPS, then CPU. "
            "Default: auto"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Checkpoint to load. Default: {CHECKPOINT_PATH}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)

    # A checkpoint contains both learned weights and the architecture settings
    # needed to construct a compatible model object.
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    print(f"using device: {describe_device(device)}")
    if "iteration" in checkpoint and "val_loss" in checkpoint:
        print(
            f"loaded checkpoint from step {checkpoint['iteration']} "
            f"with val loss {checkpoint['val_loss']:.4f}"
        )
    print(f"using tokenizer: {tokenizer.name}")

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
    # eval() disables dropout so generation uses the full learned model.
    model.eval()

    if args.prompt:
        start_tokens = tokenizer.encode(args.prompt)
    else:
        start_tokens = tokenizer.encode("\n") or [0]

    start = torch.tensor([start_tokens], dtype=torch.long, device=device)
    top_k = args.top_k if args.top_k > 0 else None
    top_p = args.top_p if args.top_p < 1.0 else None

    # Temperature controls sampling risk. Lower values prefer high-confidence
    # tokens; higher values produce more variety and more mistakes.
    # Top-k keeps sampling inside the k most likely tokens at each step.
    # Top-p keeps the smallest likely-token set whose probability mass reaches
    # p, so the candidate count adapts to model confidence.
    generated = model.generate(
        start,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=top_p,
    )[0].tolist()
    text = tokenizer.decode(generated)
    print(text)


if __name__ == "__main__":
    main()
