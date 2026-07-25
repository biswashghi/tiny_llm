# Tokenization Design

This document explains the tokenizer layer added after the character-level
Transformer started producing Sherlock-like style but still struggled with
spelling and word-level coherence.

## Why Tokenization Matters

The model predicts one token at a time. With character tokenization, a word like
`Watson` is six prediction steps:

| Step | Token |
| ---: | --- |
| 1 | `W` |
| 2 | `a` |
| 3 | `t` |
| 4 | `s` |
| 5 | `o` |
| 6 | `n` |

With BPE tokenization, frequent chunks can become single tokens or short token
sequences:

| Text | Possible BPE tokens |
| --- | --- |
| `Watson` | `Watson` |
| `Holmes looked` | `Holmes`, ` looked` |
| `investigation` | `invest`, `igation` |

That means the same 128-token context covers more story, and the model spends
less capacity relearning common spelling patterns.

## Supported Modes

| Mode | Dependency | Strength | Tradeoff |
| --- | --- | --- | --- |
| `char` | PyTorch only | Very easy to inspect and compare | Weak spelling; short effective context |
| `bpe` | Hugging Face `tokenizers` | More realistic LLM-style subword chunks | Larger vocabulary and output head |

`char` remains the default so older learning commands still behave the same.
BPE is opt-in:

```bash
python3 train.py --data data/cleaned/sherlock --tokenizer bpe --vocab-size 2000
python3 generate.py "My dear Watson"
```

## Data Flow

| Stage | Char mode | BPE mode |
| --- | --- | --- |
| Read cleaned text | Same | Same |
| Build tokenizer | Sort unique characters | Train Byte-Level BPE on corpus |
| Encode corpus | One ID per character | One ID per byte/subword token |
| Train model | Same transformer | Same transformer |
| Save checkpoint | Save `stoi` and `itos` | Save serialized tokenizer JSON |
| Generate text | Decode IDs as characters | Decode IDs with BPE decoder |

The transformer only sees integer IDs, so `model.py` does not need to know which
tokenizer produced them.

## Checkpoint Compatibility

| Checkpoint type | Generate behavior |
| --- | --- |
| Old char checkpoint | Falls back to legacy `stoi`/`itos` metadata |
| New char checkpoint | Uses explicit `tokenizer_type: char` metadata |
| New BPE checkpoint | Rebuilds tokenizer from serialized `tokenizer_json` |

BPE training also writes a sidecar tokenizer file beside the checkpoint:

```text
checkpoints/tiny_transformer.tokenizer.json
checkpoints/tiny_transformer_best.tokenizer.json
```

The sidecar is for humans and debugging. The checkpoint itself has enough
metadata for `generate.py` to work.

## Resume Training

Resuming reuses the checkpoint tokenizer and architecture instead of rebuilding
them from command-line tokenizer flags. That prevents BPE token IDs or model
shapes from drifting away from the saved weights.

```bash
python3 train.py --data data/cleaned/sherlock --resume checkpoints/tiny_transformer_best.pt --max-iters 24000
```

`--max-iters` means additional optimizer steps for this run. If the checkpoint
was saved at step `7750`, the command above finishes around step `31750`.

New checkpoints store optimizer state as well as model weights. Older
checkpoints without optimizer state still resume, but AdamW starts with fresh
optimizer moments.

## Performance Notes

Tokenization changes representation quality. The same pass also improves a few
runtime details:

| Change | Why it helps |
| --- | --- |
| Device-resident data tensors | Avoids moving every batch from CPU to accelerator |
| Vectorized batch indexing | Avoids Python loops for batch window construction |
| Reused position IDs | Avoids rebuilding identical position tensors each forward |
| Reused causal masks | Avoids rebuilding attention masks in every block call |
| `torch.inference_mode()` | Faster and stricter no-gradient eval/generation |
| Optional `--compile` | Lets PyTorch try graph compilation when supported |
| Optional `--amp` | Lets PyTorch try autocast mixed precision when supported |

`--compile` and `--amp` are intentionally opt-in. They are useful learning
experiments, but Apple Silicon MPS support can vary by PyTorch version.

## What To Compare

Run one char checkpoint and one BPE checkpoint on the same Sherlock corpus.
Compare:

| Question | Expected signal |
| --- | --- |
| Is spelling better? | BPE should produce fewer mangled words |
| Is context better? | BPE should carry names and phrases farther |
| Is training loss comparable? | BPE loss is not directly comparable to char loss |
| Is training slower? | BPE has a larger output head, so each step may cost more |
| Is sample quality better? | Prompted Sherlock outputs should feel less letter-by-letter |

Loss values across tokenizers are not apples-to-apples because the prediction
unit changed. Generated samples matter more for this comparison.

## Experiment Log

Ongoing training results, validation-loss observations, learning-rate notes,
and architecture comparisons now live in
[`experiment-log.md`](experiment-log.md).

Keeping those notes separate makes this file easier to use as the tokenizer
design reference, while the experiment log stays free to grow like a lab
notebook.
