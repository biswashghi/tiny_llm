from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable


EncodeFn = Callable[[str], list[int]]
DecodeFn = Callable[[list[int]], str]


@dataclass
class TokenizerBundle:
    """Small adapter so train.py and generate.py can share tokenizer behavior."""

    name: str
    vocab_size: int
    encode: EncodeFn
    decode: DecodeFn
    metadata: dict[str, Any]


def build_tokenizer(
    text: str,
    tokenizer_name: str,
    vocab_size: int = 2_000,
    min_frequency: int = 2,
) -> TokenizerBundle:
    if tokenizer_name == "char":
        return build_char_tokenizer(text)
    if tokenizer_name == "bpe":
        return build_bpe_tokenizer(
            text=text,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )
    raise ValueError(f"unknown tokenizer: {tokenizer_name}")


def build_char_tokenizer(text: str) -> TokenizerBundle:
    # Character tokenization is the baseline: every unique character becomes a
    # token. It is simple and debuggable, but the model must learn spelling from
    # individual letters.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(value: str) -> list[int]:
        return [stoi[ch] for ch in value]

    def decode(tokens: list[int]) -> str:
        return "".join(itos[token] for token in tokens)

    return TokenizerBundle(
        name="char",
        vocab_size=len(chars),
        encode=encode,
        decode=decode,
        metadata={
            "tokenizer_type": "char",
            "stoi": stoi,
            "itos": itos,
            "vocab_size": len(chars),
        },
    )


def build_bpe_tokenizer(
    text: str,
    vocab_size: int,
    min_frequency: int,
) -> TokenizerBundle:
    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise ImportError(
            "BPE tokenization requires the Hugging Face tokenizers package. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    # Byte-level BPE starts from bytes, then learns frequent byte-pair merges.
    # That gives the model word/subword-sized tokens without losing the ability
    # to represent unusual punctuation or spelling.
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>"],
        initial_alphabet=ByteLevel.alphabet(),
    )
    with NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as temp_file:
        temp_file.write(text)
        temp_file.flush()
        tokenizer.train([temp_file.name], trainer)

    def encode(value: str) -> list[int]:
        return tokenizer.encode(value).ids

    def decode(tokens: list[int]) -> str:
        return tokenizer.decode(tokens)

    serialized = tokenizer.to_str(pretty=True)
    return TokenizerBundle(
        name="bpe",
        vocab_size=tokenizer.get_vocab_size(),
        encode=encode,
        decode=decode,
        metadata={
            "tokenizer_type": "bpe",
            "tokenizer_json": serialized,
            "requested_vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "vocab_size": tokenizer.get_vocab_size(),
        },
    )


def tokenizer_from_checkpoint(checkpoint: dict[str, Any]) -> TokenizerBundle:
    tokenizer_type = checkpoint.get("tokenizer_type")
    if tokenizer_type == "char":
        return char_tokenizer_from_metadata(
            stoi=checkpoint["stoi"],
            itos=checkpoint["itos"],
            vocab_size=checkpoint["vocab_size"],
        )
    if tokenizer_type == "bpe":
        return bpe_tokenizer_from_json(checkpoint["tokenizer_json"])

    # Legacy checkpoints from before tokenizer metadata used stoi/itos directly.
    if "stoi" in checkpoint and "itos" in checkpoint:
        return char_tokenizer_from_metadata(
            stoi=checkpoint["stoi"],
            itos=checkpoint["itos"],
            vocab_size=checkpoint["vocab_size"],
        )

    raise ValueError("checkpoint does not contain tokenizer metadata")


def char_tokenizer_from_metadata(
    stoi: dict[str, int],
    itos: dict[int, str],
    vocab_size: int,
) -> TokenizerBundle:
    # torch.save/load may round-trip integer dictionary keys as strings in some
    # contexts, so normalize the decode map before using it.
    normalized_itos = {int(token): ch for token, ch in itos.items()}

    def encode(value: str) -> list[int]:
        unknown_chars = sorted({ch for ch in value if ch not in stoi})
        if unknown_chars:
            unknown = "".join(unknown_chars)
            raise ValueError(
                f"prompt contains characters not in vocabulary: {unknown!r}"
            )
        return [stoi[ch] for ch in value]

    def decode(tokens: list[int]) -> str:
        return "".join(normalized_itos[token] for token in tokens)

    return TokenizerBundle(
        name="char",
        vocab_size=vocab_size,
        encode=encode,
        decode=decode,
        metadata={
            "tokenizer_type": "char",
            "stoi": stoi,
            "itos": normalized_itos,
            "vocab_size": vocab_size,
        },
    )


def bpe_tokenizer_from_json(tokenizer_json: str) -> TokenizerBundle:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ImportError(
            "This checkpoint uses a BPE tokenizer. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    tokenizer = Tokenizer.from_str(tokenizer_json)

    def encode(value: str) -> list[int]:
        return tokenizer.encode(value).ids

    def decode(tokens: list[int]) -> str:
        return tokenizer.decode(tokens)

    return TokenizerBundle(
        name="bpe",
        vocab_size=tokenizer.get_vocab_size(),
        encode=encode,
        decode=decode,
        metadata={
            "tokenizer_type": "bpe",
            "tokenizer_json": tokenizer_json,
            "vocab_size": tokenizer.get_vocab_size(),
        },
    )


def write_tokenizer_sidecar(
    checkpoint_path: Path,
    tokenizer: TokenizerBundle,
) -> Path | None:
    if tokenizer.name != "bpe":
        return None

    sidecar_path = checkpoint_path.with_suffix(".tokenizer.json")
    sidecar_path.write_text(tokenizer.metadata["tokenizer_json"], encoding="utf-8")
    return sidecar_path
