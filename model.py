import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Predicts the next token using only the current token."""

    def __init__(self, vocab_size: int):
        super().__init__()
        # Baseline lookup table: each current token maps directly to next-token
        # logits. There is no context, hidden state, or attention here.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            batch_size, context_length, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * context_length, vocab_size)
            targets_flat = targets.view(batch_size * context_length)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            logits = logits / temperature
            logits = filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


class TinyTransformerLanguageModel(nn.Module):
    """A small causal Transformer that predicts the next token from recent context."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        tie_weights: bool = False,
    ):
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")

        self.context_length = context_length
        self.activation = activation
        self.tie_weights = tie_weights
        self.register_buffer(
            "position_ids",
            torch.arange(context_length),
            persistent=False,
        )

        # Embedding tables are learned lookup tables. For a software analogy,
        # this turns a sparse enum value into a dense feature vector the rest of
        # the model can mutate and compare.
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)

        # Attention sees a set of vectors; position embeddings inject ordering
        # information so "a" at position 3 differs from "a" at position 80.
        self.position_embedding_table = nn.Embedding(context_length, embedding_dim)

        # A stack of Transformer blocks gives the model multiple rounds of
        # "read prior context, combine signals, rewrite each token's features."
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    context_length=context_length,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # The output head is the classifier. At every position, it maps the
        # final hidden vector to one score per possible next token.
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=not tie_weights)
        self.apply(self._init_weights)
        if tie_weights:
            # Weight tying reuses the token embedding matrix as the output
            # classifier. Input token vectors and output token classifiers become
            # two views of the same learned table, a common language-model trick.
            self.output_head.weight = self.token_embedding_table.weight

    def _init_weights(self, module: nn.Module):
        # Small Transformer-style initialization keeps early logits in a sane
        # range. This matters especially for weight tying, where embedding rows
        # are reused directly as output classifiers.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx is an integer tensor of token IDs: [batch, context].
        # Example with current config: [32, 128].
        batch_size, context_length = idx.shape
        if context_length > self.context_length:
            raise ValueError(
                f"Cannot process {context_length} tokens with context length "
                f"{self.context_length}."
            )

        token_embeddings = self.token_embedding_table(idx)
        positions = self.position_ids[:context_length]
        position_embeddings = self.position_embedding_table(positions)

        # Broadcasting adds the same [context, embedding] position table to each
        # batch row, producing dense token state: [batch, context, embedding].
        x = self.dropout(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)

        # logits are raw, unnormalized scores. Softmax is intentionally not used
        # during training because cross_entropy combines log-softmax + NLL loss
        # in one numerically stable operation.
        logits = self.output_head(self.final_layer_norm(x))

        loss = None
        if targets is not None:
            # Cross-entropy expects a flat list of examples:
            #   predictions: [batch * context, vocab]
            #   labels:      [batch * context]
            # Each token position in the batch becomes one training example.
            batch_size, context_length, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * context_length, vocab_size)
            targets_flat = targets.reshape(batch_size * context_length)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        for _ in range(max_new_tokens):
            # Generation is autoregressive: run the model on the recent context,
            # sample one next token, append it, then repeat.
            idx_cond = idx[:, -self.context_length :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # Lower temperature sharpens the probability distribution; higher
            # temperature flattens it and makes sampling more surprising.
            logits = logits / temperature

            # Top-k and top-p are decoding heuristics: they change generation
            # behavior without changing what the model learned during training.
            logits = filter_logits(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


def filter_logits(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    return logits


def apply_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    top_k = min(top_k, logits.shape[-1])
    values, _ = torch.topk(logits, top_k)
    cutoff = values[:, [-1]]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None:
        return logits
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and less than or equal to 1")

    # Nucleus sampling sorts candidates by confidence, keeps the smallest group
    # whose probability mass reaches top_p, and masks the rest. Unlike top-k,
    # the candidate count adapts to how certain the model is at this step.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove_mask = cumulative_probs > top_p
    sorted_remove_mask[:, 1:] = sorted_remove_mask[:, :-1].clone()
    sorted_remove_mask[:, 0] = False

    remove_mask = torch.zeros_like(sorted_remove_mask)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove_mask)
    return logits.masked_fill(remove_mask, float("-inf"))


class TransformerBlock(nn.Module):
    """One causal self-attention block plus one feedforward block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        if activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "gelu":
            activation_layer = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

        # Multi-head attention runs several smaller attention mechanisms in
        # parallel, letting different heads specialize in different patterns.
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        # The MLP is position-wise: it does not mix tokens with each other.
        # Attention handles communication; this network transforms each token's
        # features after that communication step.
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        # Pre-norm layout: normalize before each sublayer. This tends to make
        # deeper Transformer stacks easier to optimize.
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_attention_mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, context_length, _ = x.shape

        # The causal mask is an upper-triangular "do not read" matrix. It keeps
        # training honest: position 10 may use positions <= 10, never 11+.
        attention_mask = self.causal_attention_mask[:context_length, :context_length]

        # Residual connections are additive patches to state. They let a block
        # contribute an update without destroying the representation it received.
        normalized_x = self.layer_norm_1(x)
        attention_output, _ = self.attention(
            normalized_x,
            normalized_x,
            normalized_x,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = x + self.dropout(attention_output)

        # Attention mixes information across token positions; the feedforward
        # network then performs a local nonlinear transform at every position.
        x = x + self.dropout(self.feed_forward(self.layer_norm_2(x)))
        return x
