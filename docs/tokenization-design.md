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

## Latest Training Notes

The Sherlock BPE run improved substantially with more training:

| Phase | Best validation loss | What changed |
| --- | ---: | --- |
| Initial BPE run | about `3.76` | Readable dialogue and better spelling |
| Resumed long run | about `3.60` | Stronger Sherlock style and longer grammatical runs |

The important lesson was not just "train longer." The validation curve kept
improving, but it also bounced. After step `22000`, the best checkpoint was
better than the final checkpoint. That taught two practical habits:

| Habit | Reason |
| --- | --- |
| Save best validation checkpoints | The final step is not guaranteed to be best |
| Decay the learning rate late in training | Smaller updates can refine instead of bouncing around |

BPE also changed what sample quality means. The model now spells much better,
but it still does not reason. It has learned local Doyle-like texture: dialogue,
names, and common phrasing. Plot logic and stable facts require more capacity,
longer context, better objectives, or a much larger pretrained model.

## Learning-Rate Decay

Training now defaults to cosine decay for each invocation:

```bash
python3 train.py --data data/cleaned/sherlock \
  --resume checkpoints/tiny_transformer_best.pt \
  --max-iters 8000 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --lr-decay cosine
```

`--max-iters` is the length of the current run, so the cosine schedule decays
over those additional steps. Use `--lr-decay none` to return to constant
learning rate behavior.

## Evaluation And Next Experiments

The next workflow is deliberately more disciplined:

```bash
python3 evaluate.py --data data/cleaned/sherlock \
  --checkpoint checkpoints/tiny_transformer_best.pt \
  --save
```

That creates a repeatable report with measured train/validation loss, parameter
count, checkpoint metadata, and the same prompt samples. Use it before and after
each experiment.

The longer-context experiment was slower and worse early on, which is useful
evidence: attention cost rises quickly with context length. The next cheaper
architecture experiment is to keep context at `128` and increase depth:

```bash
python3 train.py --data data/cleaned/sherlock \
  --tokenizer bpe \
  --vocab-size 2000 \
  --context-length 128 \
  --num-layers 6 \
  --batch-size 16 \
  --checkpoint checkpoints/tiny_transformer_layers6.pt \
  --best-checkpoint checkpoints/tiny_transformer_layers6_best.pt
```

Use a separate checkpoint path because changing layer count changes the model
shape. Resume is for continuing a compatible checkpoint; architecture
experiments start a new checkpoint lineage.

The 6-layer run got close to the 3-layer best but did not clearly beat it. The
next experiment keeps context at `128`, uses `GELU` in the MLP, and ties the
token embedding table to the output head:

```bash
python3 train.py --data data/cleaned/sherlock \
  --tokenizer bpe \
  --vocab-size 2000 \
  --context-length 128 \
  --num-layers 6 \
  --activation gelu \
  --tie-weights \
  --batch-size 16 \
  --checkpoint checkpoints/tiny_transformer_gelu_tied.pt \
  --best-checkpoint checkpoints/tiny_transformer_gelu_tied_best.pt
```

This tests a more modern Transformer block without increasing context length.
Old checkpoints remain compatible because missing metadata defaults to `relu`
and untied weights.
