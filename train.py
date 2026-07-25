import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from devices import DEVICE_CHOICES, choose_device, describe_device
from model import TinyTransformerLanguageModel
from tokenization import build_tokenizer, tokenizer_from_checkpoint, write_tokenizer_sidecar


BATCH_SIZE = 32
CONTEXT_LENGTH = 128
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.1
ACTIVATION = "gelu"
TIE_WEIGHTS = True
MAX_ITERS = 8_000
EVAL_INTERVAL = 250
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-4
GRAD_CLIP = 1.0
SAMPLE_TEMPERATURE = 0.8
SAMPLE_TOP_K = 20
SAMPLE_TOP_P = 0.95

ROOT = Path(__file__).parent
DEFAULT_DATA_PATH = ROOT / "data" / "cleaned"
CHECKPOINT_PATH = ROOT / "checkpoints" / "tiny_transformer.pt"
BEST_CHECKPOINT_PATH = ROOT / "checkpoints" / "tiny_transformer_best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the tiny language model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Cleaned training file or folder. Default: {DEFAULT_DATA_PATH}",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("char", "bpe"),
        default="char",
        help="Tokenization mode. Default: char",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=2_000,
        help="Target BPE vocabulary size. Used only with --tokenizer bpe.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum BPE merge frequency. Used only with --tokenizer bpe.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Experimentally wrap the model with torch.compile when available.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Experimentally use torch.autocast during forward/backward passes.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume model weights/tokenizer from a checkpoint.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=MAX_ITERS,
        help=f"Number of optimizer steps to run. Default: {MAX_ITERS}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Training batch size. Default: {BATCH_SIZE}",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=20,
        help="Number of random batches to average for each loss estimate. Default: 20",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help=f"Context tokens visible to the model. Default: {CONTEXT_LENGTH}",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM,
        help=f"Transformer embedding width. Default: {EMBEDDING_DIM}",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=NUM_HEADS,
        help=f"Attention heads per block. Default: {NUM_HEADS}",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=NUM_LAYERS,
        help=f"Transformer blocks. Default: {NUM_LAYERS}",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT,
        help=f"Training dropout probability. Default: {DROPOUT}",
    )
    parser.add_argument(
        "--activation",
        choices=("relu", "gelu"),
        default=ACTIVATION,
        help=f"Feedforward activation. Default: {ACTIVATION}",
    )
    parser.add_argument(
        "--tie-weights",
        dest="tie_weights",
        action="store_true",
        default=TIE_WEIGHTS,
        help="Tie token embedding and output-head weights. Default: enabled",
    )
    parser.add_argument(
        "--no-tie-weights",
        dest="tie_weights",
        action="store_false",
        help="Disable token embedding/output-head weight tying.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Starting learning rate. Default: {LEARNING_RATE}",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=MIN_LEARNING_RATE,
        help=f"Final learning rate for cosine decay. Default: {MIN_LEARNING_RATE}",
    )
    parser.add_argument(
        "--lr-decay",
        choices=("cosine", "none"),
        default="cosine",
        help="Learning-rate schedule for this run. Default: cosine",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=GRAD_CLIP,
        help=(
            "Clip gradients to this max norm before optimizer.step(). "
            f"Use 0 to disable. Default: {GRAD_CLIP}"
        ),
    )
    parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default="auto",
        help=(
            "Training device. 'auto' prefers CUDA, then Apple Silicon MPS, then CPU. "
            "Default: auto"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=f"Final checkpoint output path. Default: {CHECKPOINT_PATH}",
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=BEST_CHECKPOINT_PATH,
        help=f"Best-validation checkpoint output path. Default: {BEST_CHECKPOINT_PATH}",
    )
    return parser.parse_args()


def load_training_text(data_path: Path) -> tuple[str, list[Path]]:
    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        files = sorted(data_path.glob("*.txt"))
    else:
        raise FileNotFoundError(f"data path does not exist: {data_path}")

    if not files:
        raise FileNotFoundError(f"no .txt files found in {data_path}")

    # Separate files with blank lines so the model does not learn to run the end
    # of one source directly into the beginning of another.
    text = "\n\n".join(file.read_text(encoding="utf-8").strip() for file in files)
    return text.strip() + "\n", files


def get_batch(data: torch.Tensor, context_length: int, batch_size: int):
    # Randomly sample windows from the 1D token stream. x is the context the
    # model sees; y is the same window shifted left by one token.
    #
    # Example:
    #   text: "hello"
    #   x:    "hell"
    #   y:    "ello"
    #
    # This vectorized version avoids a Python loop by building an index matrix:
    # each row is start_position + [0, 1, 2, ..., context_length].
    starts = torch.randint(
        len(data) - context_length,
        (batch_size, 1),
        device=data.device,
    )
    offsets = torch.arange(context_length + 1, device=data.device)
    windows = data[starts + offsets]
    x = windows[:, :-1]
    y = windows[:, 1:]
    return x, y


@torch.inference_mode()
def estimate_loss(
    model: TinyTransformerLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    context_length: int,
    batch_size: int,
    eval_batches: int,
    amp_enabled: bool,
    device: str,
):
    # Evaluation disables gradient tracking and dropout. Averaging multiple
    # random batches gives a less noisy estimate than checking a single batch.
    model.eval()
    losses = {}

    for split, data in (("train", train_data), ("val", val_data)):
        split_losses = torch.zeros(eval_batches)
        for step in range(eval_batches):
            x, y = get_batch(data, context_length, batch_size)
            with autocast_context(amp_enabled, device):
                _, loss = model(x, y)
            split_losses[step] = loss.item()
        losses[split] = split_losses.mean().item()

    model.train()
    return losses


def build_checkpoint(
    model: TinyTransformerLanguageModel,
    optimizer_state_dict: dict | None,
    tokenizer_metadata: dict,
    vocab_size: int,
    context_length: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    activation: str,
    tie_weights: bool,
    learning_rate: float,
    min_learning_rate: float,
    lr_decay: str,
    grad_clip: float,
    iteration: int,
    train_loss: float,
    val_loss: float,
):
    # Save model metadata with the weights so generate.py can rebuild the same
    # architecture before loading learned parameters.
    checkpoint = {
        "model_type": "tiny_transformer",
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "context_length": context_length,
        "embedding_dim": embedding_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "activation": activation,
        "tie_weights": tie_weights,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "lr_decay": lr_decay,
        "grad_clip": grad_clip,
        "iteration": iteration,
        "train_loss": train_loss,
        "val_loss": val_loss,
        **tokenizer_metadata,
    }
    if optimizer_state_dict is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict
    return checkpoint


def autocast_context(enabled: bool, device: str):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device)


def maybe_compile_model(model: TinyTransformerLanguageModel, enabled: bool):
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        print("torch.compile requested but this PyTorch build does not support it")
        return model

    try:
        compiled_model = torch.compile(model)
    except Exception as exc:
        print(f"torch.compile requested but could not start: {exc}")
        return model

    print("using torch.compile: enabled")
    return compiled_model


def get_learning_rate(
    local_iteration: int,
    max_iters: int,
    learning_rate: float,
    min_learning_rate: float,
    lr_decay: str,
) -> float:
    if lr_decay == "none":
        return learning_rate

    # Cosine decay starts at the requested learning rate and eases down toward
    # min_learning_rate. Near the end of a run, this makes smaller updates so the
    # optimizer is less likely to bounce around a good validation basin.
    progress = local_iteration / max(1, max_iters - 1)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine_weight * (learning_rate - min_learning_rate)


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def load_resume_checkpoint(resume_path: Path) -> dict | None:
    if resume_path is None:
        return None
    if not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    return torch.load(resume_path, map_location="cpu")


def main():
    args = parse_args()
    if args.max_iters < 0:
        raise ValueError("--max-iters must be greater than or equal to 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be greater than 0")
    if args.context_length <= 0:
        raise ValueError("--context-length must be greater than 0")
    if args.embedding_dim <= 0:
        raise ValueError("--embedding-dim must be greater than 0")
    if args.num_heads <= 0:
        raise ValueError("--num-heads must be greater than 0")
    if args.num_layers <= 0:
        raise ValueError("--num-layers must be greater than 0")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be greater than or equal to 0 and less than 1")
    if args.embedding_dim % args.num_heads != 0:
        raise ValueError("--embedding-dim must be divisible by --num-heads")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0")
    if args.min_learning_rate <= 0:
        raise ValueError("--min-learning-rate must be greater than 0")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("--min-learning-rate must be less than or equal to --learning-rate")
    if args.grad_clip < 0:
        raise ValueError("--grad-clip must be greater than or equal to 0")

    device = choose_device(args.device)
    text, data_files = load_training_text(args.data)
    resume_checkpoint = load_resume_checkpoint(args.resume)
    if resume_checkpoint is None:
        tokenizer = build_tokenizer(
            text=text,
            tokenizer_name=args.tokenizer,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        context_length = args.context_length
        embedding_dim = args.embedding_dim
        num_heads = args.num_heads
        num_layers = args.num_layers
        dropout = args.dropout
        activation = args.activation
        tie_weights = args.tie_weights
        start_iteration = 0
        best_val_loss = float("inf")
        best_iteration = 0
    else:
        # Resume uses the checkpoint tokenizer and architecture. That keeps the
        # integer token IDs and model shapes aligned with the saved weights.
        tokenizer = tokenizer_from_checkpoint(resume_checkpoint)
        context_length = resume_checkpoint["context_length"]
        embedding_dim = resume_checkpoint["embedding_dim"]
        num_heads = resume_checkpoint["num_heads"]
        num_layers = resume_checkpoint.get("num_layers", 1)
        dropout = resume_checkpoint.get("dropout", 0.0)
        activation = resume_checkpoint.get("activation", "relu")
        tie_weights = resume_checkpoint.get("tie_weights", False)
        start_iteration = resume_checkpoint.get("iteration", 0)
        best_val_loss = resume_checkpoint.get("val_loss", float("inf"))
        best_iteration = start_iteration

    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded, dtype=torch.long, device=device)
    amp_enabled = args.amp and device != "cpu"

    print(
        f"training on {len(text):,} characters, "
        f"{len(encoded):,} {tokenizer.name} tokens, "
        f"vocab size {tokenizer.vocab_size:,}"
    )
    print(f"loaded {len(data_files)} cleaned data file(s) from {args.data}")
    print(f"using device: {describe_device(device)}")
    print(
        "model config: "
        f"context {context_length}, embedding {embedding_dim}, "
        f"heads {num_heads}, layers {num_layers}, dropout {dropout}, "
        f"activation {activation}, tie_weights {tie_weights}"
    )
    print(f"batch size: {args.batch_size}, eval batches: {args.eval_batches}")
    if resume_checkpoint is not None:
        print(f"resuming from {args.resume} at step {start_iteration}")
        print("resume uses checkpoint architecture; architecture CLI flags are ignored")
    if args.amp and not amp_enabled:
        print("amp requested but disabled because the selected device is cpu")
    elif amp_enabled:
        print("using amp: enabled")
    if args.lr_decay == "cosine":
        print(
            "using learning rate: "
            f"cosine decay from {args.learning_rate:g} to {args.min_learning_rate:g}"
        )
    else:
        print(f"using learning rate: constant {args.learning_rate:g}")
    if args.grad_clip > 0:
        print(f"using gradient clipping: max norm {args.grad_clip:g}")
    else:
        print("using gradient clipping: disabled")

    split_idx = int(0.9 * len(data))
    # The validation split is held out from optimizer updates. If train loss
    # falls while val loss rises, the model is memorizing instead of generalizing.
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    raw_model = TinyTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        activation=activation,
        tie_weights=tie_weights,
    ).to(device)
    if resume_checkpoint is not None:
        raw_model.load_state_dict(resume_checkpoint["model_state_dict"])

    model = maybe_compile_model(raw_model, args.compile)
    # AdamW is the optimizer that applies gradient updates. Compared with plain
    # SGD, it adapts step sizes per parameter and handles weight decay cleanly.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if resume_checkpoint is not None and "optimizer_state_dict" in resume_checkpoint:
        try:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            print("resumed optimizer state")
        except ValueError as exc:
            print(f"could not resume optimizer state: {exc}")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for local_iteration in range(args.max_iters):
        iteration = start_iteration + local_iteration
        current_learning_rate = get_learning_rate(
            local_iteration=local_iteration,
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            lr_decay=args.lr_decay,
        )
        set_optimizer_learning_rate(optimizer, current_learning_rate)
        if iteration % EVAL_INTERVAL == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                context_length,
                args.batch_size,
                args.eval_batches,
                amp_enabled,
                device,
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_iteration = iteration
                torch.save(
                    build_checkpoint(
                        model=raw_model,
                        optimizer_state_dict=optimizer.state_dict(),
                        tokenizer_metadata=tokenizer.metadata,
                        vocab_size=tokenizer.vocab_size,
                        context_length=context_length,
                        embedding_dim=embedding_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        dropout=dropout,
                        activation=activation,
                        tie_weights=tie_weights,
                        learning_rate=current_learning_rate,
                        min_learning_rate=args.min_learning_rate,
                        lr_decay=args.lr_decay,
                        grad_clip=args.grad_clip,
                        iteration=iteration,
                        train_loss=losses["train"],
                        val_loss=losses["val"],
                    ),
                    args.best_checkpoint,
                )
                best_note = " best"
            else:
                best_note = ""

            print(
                f"step {iteration}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"lr {current_learning_rate:.2e}{best_note}"
            )

        x, y = get_batch(train_data, context_length, args.batch_size)
        with autocast_context(amp_enabled, device):
            _, loss = model(x, y)

        # Standard training step:
        #   1. clear old gradients
        #   2. backprop computes d(loss)/d(parameter)
        #   3. optionally clip huge gradients so one strange batch cannot make
        #      an outsized parameter update
        #   4. optimizer nudges parameters to reduce future loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    losses = estimate_loss(
        model,
        train_data,
        val_data,
        context_length,
        args.batch_size,
        args.eval_batches,
        amp_enabled,
        device,
    )
    final_iteration = start_iteration + args.max_iters
    print(
        f"final: train loss {losses['train']:.4f}, "
        f"val loss {losses['val']:.4f}"
    )
    if losses["val"] < best_val_loss:
        best_val_loss = losses["val"]
        best_iteration = final_iteration
        final_learning_rate = get_learning_rate(
            local_iteration=max(args.max_iters - 1, 0),
            max_iters=max(args.max_iters, 1),
            learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            lr_decay=args.lr_decay,
        )
        torch.save(
            build_checkpoint(
                model=raw_model,
                optimizer_state_dict=optimizer.state_dict(),
                tokenizer_metadata=tokenizer.metadata,
                vocab_size=tokenizer.vocab_size,
                context_length=context_length,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                activation=activation,
                tie_weights=tie_weights,
                learning_rate=final_learning_rate,
                min_learning_rate=args.min_learning_rate,
                lr_decay=args.lr_decay,
                grad_clip=args.grad_clip,
                iteration=final_iteration,
                train_loss=losses["train"],
                val_loss=losses["val"],
            ),
            args.best_checkpoint,
        )

    torch.save(
        build_checkpoint(
            model=raw_model,
            optimizer_state_dict=optimizer.state_dict(),
            tokenizer_metadata=tokenizer.metadata,
            vocab_size=tokenizer.vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            tie_weights=tie_weights,
            learning_rate=get_learning_rate(
                local_iteration=max(args.max_iters - 1, 0),
                max_iters=max(args.max_iters, 1),
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                lr_decay=args.lr_decay,
            ),
            min_learning_rate=args.min_learning_rate,
            lr_decay=args.lr_decay,
            grad_clip=args.grad_clip,
            iteration=final_iteration,
            train_loss=losses["train"],
            val_loss=losses["val"],
        ),
        args.checkpoint,
    )
    sidecar_path = write_tokenizer_sidecar(args.checkpoint, tokenizer)
    best_sidecar_path = write_tokenizer_sidecar(args.best_checkpoint, tokenizer)

    model.eval()
    raw_model.eval()
    start_tokens = tokenizer.encode("\n") or [0]
    start = torch.tensor([start_tokens], dtype=torch.long, device=device)
    generated = raw_model.generate(
        start,
        max_new_tokens=300,
        temperature=SAMPLE_TEMPERATURE,
        top_k=SAMPLE_TOP_K,
        top_p=SAMPLE_TOP_P,
    )[0].tolist()
    print("\nSample:")
    print(tokenizer.decode(generated))
    print(f"\nSaved checkpoint to {args.checkpoint}")
    print(
        f"Saved best checkpoint to {args.best_checkpoint} "
        f"(step {best_iteration}, val loss {best_val_loss:.4f})"
    )
    if sidecar_path is not None:
        print(f"Saved tokenizer to {sidecar_path}")
    if best_sidecar_path is not None:
        print(f"Saved best tokenizer to {best_sidecar_path}")


if __name__ == "__main__":
    main()
