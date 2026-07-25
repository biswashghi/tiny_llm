# Experiment Log

This is the lab notebook for the tiny LLM project. The README explains how to
run the project; this file records what each learning phase taught.

## Learning Timeline

1. **Bigram baseline**

   Started with `BigramLanguageModel`, a learned lookup table that predicts the
   next character from only the current character.

   ```text
   current character -> next character
   ```

   Taught: tensors, embeddings, logits, cross-entropy loss, and autoregressive
   generation.

2. **Validation loss**

   Added a train/validation split and periodic loss estimates.

   ```text
   train loss = fit to training examples
   val loss   = prediction quality on held-out text
   ```

   Taught: lower training loss alone does not mean the model is better.

3. **Tiny Transformer**

   Replaced the bigram-only path with `TinyTransformerLanguageModel`: token
   embeddings, position embeddings, causal self-attention, feedforward MLP,
   LayerNorm, residual connections, and dropout.

   ```text
   recent token window -> attention over context -> next token
   ```

   Taught: GPT-style models use context instead of one-token lookup.

4. **Cleaner training text**

   Moved raw text cleanup into `scripts/clean_input.py`. Raw files live in
   `data/source/`; cleaned model-ready files live in `data/cleaned/`.

   Taught: small models are sensitive to data distribution. More text is not
   automatically better if the text contains unrelated styles or formats.

5. **Apple Silicon acceleration**

   Updated device selection to prefer CUDA, then Apple Silicon MPS, then CPU.

   ```text
   cuda -> mps -> cpu
   ```

   Taught: Python drives the loop, but PyTorch runs tensor operations on
   optimized backends.

6. **Larger training config**

   Increased context length and batch coverage before changing architecture.

   ```text
   context length: 32 -> 64 -> 128
   batch coverage: batch_size * context_length
   ```

   Taught: each batch contains many next-token prediction examples, and context
   length controls how far back the model can look.

7. **Stacked Transformer blocks**

   Refactored the model into reusable `TransformerBlock` layers and moved to a
   3-block, 128-dimensional model.

   ```text
   embedding -> block -> block -> block -> output
   ```

   Taught: depth gives multiple rounds to read, mix, and refine context.

8. **Temperature, top-k, and top-p sampling**

   Added generation controls so sampling can be less repetitive than greedy
   decoding but less chaotic than sampling from every possible token.

   ```text
   logits -> temperature -> top-k/top-p filters -> softmax -> sample
   ```

   Taught: model training and decoding strategy are separate parts of language
   generation.

9. **Prompted generation**

   Added command-line prompts so generation can start from user-provided text.

   ```text
   prompt text -> encode tokens -> model continues from prompt
   ```

   Taught: autoregressive models generate continuations conditioned on context.

10. **Best validation checkpoint**

    Training saves both the final model and the model with the lowest validation
    loss.

    ```text
    checkpoints/tiny_transformer.pt       # final step
    checkpoints/tiny_transformer_best.pt  # best validation loss
    ```

    Taught: the best model is often selected by validation performance, not the
    final training step.

11. **Byte-Level BPE tokenization**

    Added Hugging Face `tokenizers` support while keeping character tokenization
    as the default baseline.

    ```text
    characters -> bytes/subword chunks -> token IDs
    ```

    Taught: modern language models usually predict chunks, not individual
    characters. BPE greatly improved spelling and local fluency.

12. **Checkpoint resume**

    Added `--resume` and `--max-iters` so longer training can continue from a
    saved checkpoint instead of starting over.

    ```text
    checkpoint -> restore model/tokenizer -> continue optimizer steps
    ```

    Taught: a checkpoint needs tokenizer, architecture, model weights, and
    optimizer state for a clean continuation.

13. **Longer BPE training**

    Resumed the Sherlock BPE checkpoint from step `7750` to around step `24000`.
    Validation improved from about `3.76` to a new best near `3.60`, and samples
    became more readable.

    ```text
    more steps -> better spelling/style -> slower, noisier validation gains
    ```

    Taught: once a model is in the right neighborhood, validation can bounce
    while the long-term trend improves. Best checkpoint saving matters.

14. **Learning-rate decay**

    Added cosine learning-rate decay so training can start with larger updates
    and gradually shift toward smaller adjustments.

    ```text
    lr 1e-3 -> cosine decay -> lr 1e-4
    ```

    Taught: after broad patterns are learned, keeping the step size too high can
    bounce around a good validation basin.

15. **Repeatable evaluation**

    Added `evaluate.py` so checkpoints can be compared with the same data split,
    deterministic random batches, parameter counts, and prompt samples.

    ```text
    checkpoint -> measured losses + samples -> runs/eval_*.md
    ```

    Taught: ML progress needs repeatable measurement, not just vibes from one
    sample.

16. **Configurable architecture experiments**

    Added CLI knobs for batch size, context length, embedding width, heads,
    layers, activation, weight tying, and dropout.

    ```text
    baseline checkpoint -> evaluate -> train one variant -> evaluate again
    ```

    Taught: architecture changes should be explicit experiments with separate
    checkpoint paths.

17. **Longer context experiment**

    Tried BPE context length `256`. It trained much slower and was worse early
    than the context `128` champion.

    ```text
    ctx128 -> ctx256
    attention cost rises quickly
    ```

    Taught: longer context is not free. Attention cost grows roughly
    quadratically with sequence length, and the model may need much more
    training to use the extra context.

18. **Deeper Transformer experiment**

    Tried 6 Transformer blocks at context `128`. A fresh run reached about
    `3.79`; resume + decay improved it to about `3.62`, close to but not
    clearly better than the 3-layer best near `3.60`.

    ```text
    3 blocks at ctx128 -> 6 blocks at ctx128
    ```

    Taught: extra layers can help, but more capacity needs more training and is
    not automatically a better efficient model.

19. **GELU + weight tying**

    Added a GPT-like `GELU` feedforward activation and optional embedding/output
    weight tying. A first 6-layer GELU+tied run reached about `3.72`, behind the
    6-layer ReLU/untied run at the same stage and behind the best 3-layer model.

    ```text
    ReLU MLP -> GELU MLP
    separate embedding/output tables -> tied token table
    ```

    Taught: tied weights are common in larger language models, but can constrain
    a tiny model. The next clean ablation is `GELU` without weight tying.

20. **GELU without weight tying**

    Ran the clean ablation: same 6-layer BPE setup, same context length, same
    batch size, same constant `1e-3` learning rate, but with `--no-tie-weights`.
    It reached about `3.68`, beating the GELU+tied run but still trailing the
    6-layer ReLU/untied result and the 3-layer champion.

    ```text
    GELU + tied weights    -> about 3.72
    GELU + untied weights  -> about 3.68
    ReLU + untied weights  -> about 3.62
    ```

    Taught: weight tying was probably hurting this tiny model. GELU alone was
    better than GELU+tied, but this experiment does not show GELU beating the
    original ReLU block at this scale.

21. **GELU untied resume with cosine decay**

    Resumed `checkpoints/tiny_transformer_gelu_best.pt` from step `7750` and
    trained another `8000` steps with cosine decay from `1e-3` to `1e-4`. The
    run reached a new best validation loss around `3.53` at step `14000`.

    ```text
    GELU untied fresh best   -> 3.6840
    GELU untied resumed best -> 3.5331
    ```

    Taught: this model was undertrained at `8000` steps. The architecture did
    not look like a winner until it got a second training phase with a decaying
    learning rate. This is now stronger evidence for the 6-layer GELU/untied
    setup than for the older 3-layer champion.

22. **Gradient clipping added late**

    Resumed the current champion from step `14000` with a lower cosine schedule
    from `5e-4` to `5e-5` and `--grad-clip 1.0`. Early validation estimates
    hovered around `3.54` to `3.59` and did not beat the saved `3.5331`
    checkpoint.

    ```text
    saved best before run -> 3.5331
    resumed with clipping -> roughly 3.54-3.59 early
    ```

    Taught: gradient clipping is not a magic improvement when added to an
    already-trained checkpoint near a plateau. It only affects future gradient
    updates. The cleaner test is to use clipping from the beginning of a fresh
    run, where it can shape the entire optimization path.

## Current Findings

| Experiment | Best observed validation loss | Status |
| --- | ---: | --- |
| 6-layer BPE, context 128, GELU untied + cosine resume | `3.5331` | Current champion |
| Same checkpoint + late grad clip | about `3.54-3.59` early | Did not beat champion |
| 3-layer BPE, context 128 | about `3.60` | Previous champion |
| 6-layer BPE, context 128, ReLU untied | about `3.62` | Promising, not champion |
| 6-layer BPE, context 128, GELU untied fresh | `3.6840` | Needed more training |
| 6-layer BPE, context 128, GELU tied | about `3.72` | Weight tying likely too restrictive |
| 3-layer BPE, context 256 | about `3.95` early | Too slow and worse early |

## Latest Run Notes

The latest completed winning run resumed the 6-layer GELU/untied checkpoint
with Sherlock BPE data, context `128`, batch size `32`, and cosine
learning-rate decay from `1e-3` to `1e-4`.

```text
best checkpoint: checkpoints/tiny_transformer_gelu_best.pt
best step:       14000
best val loss:   3.5331
measured val:    3.5472
final val loss:  3.5468
```

The sample is qualitatively readable in short bursts: Doyle-like dialogue,
recognizable Holmes/Watson texture, and fewer broken spellings than the early
character models. It still drifts semantically and repeats vague case-language:
the model can imitate the surface of a case better than it can maintain the
facts of one.

The useful conclusion changed after the resume. Removing weight tying helped,
and giving the untied GELU model more training with a decaying learning rate
helped a lot. The best current setup is now 6 layers, context `128`, BPE,
`GELU`, untied embeddings/output head, and cosine-decayed resume training.

The follow-up attempt to add gradient clipping during another resume did not
show improvement over the saved best. That does not make clipping useless; it
means clipping should be tested as part of the optimization recipe from the
beginning, not bolted onto a near-plateau checkpoint.

## Next Ablation

Start a fresh, slightly larger model with gradient clipping enabled from step
zero. Keep context at `128` to avoid the quadratic attention slowdown, but widen
the hidden state from `128` to `192` and use `6` heads so each head still has a
clean width of `32`.

```bash
python3 train.py --data data/cleaned/sherlock \
  --tokenizer bpe \
  --vocab-size 2000 \
  --context-length 128 \
  --embedding-dim 192 \
  --num-heads 6 \
  --num-layers 6 \
  --activation gelu \
  --no-tie-weights \
  --batch-size 16 \
  --max-iters 8000 \
  --learning-rate 1e-3 \
  --min-learning-rate 1e-4 \
  --lr-decay cosine \
  --grad-clip 1.0 \
  --checkpoint checkpoints/tiny_transformer_wide192_gelu.pt \
  --best-checkpoint checkpoints/tiny_transformer_wide192_gelu_best.pt
```

Compare it against:

```bash
python3 evaluate.py --data data/cleaned/sherlock \
  --checkpoint checkpoints/tiny_transformer_best.pt \
  --save

python3 evaluate.py --data data/cleaned/sherlock \
  --checkpoint checkpoints/tiny_transformer_gelu_best.pt \
  --save

python3 evaluate.py --data data/cleaned/sherlock \
  --checkpoint checkpoints/tiny_transformer_wide192_gelu_best.pt \
  --save
```
