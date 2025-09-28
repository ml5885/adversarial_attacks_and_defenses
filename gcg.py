from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

import math
import gc
import functools
import inspect

import torch
from torch import Tensor

import transformers
from transformers import set_seed, PreTrainedTokenizerBase

INIT_CHARS = [
    ".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}",
    "@", "#", "$", "%", "&", "*",
    "w", "x", "y", "z",
]

def get_nonascii_toks(tokenizer, device="cpu"):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    nonascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            nonascii_toks.append(i)
    
    if tokenizer.bos_token_id is not None:
        nonascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        nonascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        nonascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        nonascii_toks.append(tokenizer.unk_token_id)
    
    return torch.tensor(nonascii_toks, device=device)

def mellowmax(t: Tensor, alpha=1.0, dim=-1):
   return 1.0 / alpha * (torch.logsumexp(alpha * t, dim=dim) - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device)))

# borrowed from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L69
def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

# modified from https://github.com/huggingface/accelerate/blob/85a75d4c3d0deffde2fc8b917d9b1ae1cb580eb2/src/accelerate/utils/memory.py#L87
def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions related to out-of-memory or
    CUDNN, the batch size is cut in half and passed to `function`

    `function` must take in a `batch_size` parameter as its first argument.

    Args:
        function (`callable`, *optional*):
            A function to wrap
        starting_batch_size (`int`, *optional*):
            The batch size to try and fit into memory

    Example:

    ```python
    >>> from utils import find_executable_batch_size


    >>> @find_executable_batch_size(starting_batch_size=128)
    ... def train(batch_size, model, optimizer):
    ...     ...


    >>> train(model, optimizer)
    ```
    """
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        # Guard against user error
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator

def configure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Checks if the (Hugging Face) tokenizer has a padding token and sets it if not present.

    Borrowed from https://github.com/EleutherAI/lm-evaluation-harness/blob/5c006ed417a2f4d01248d487bcbd493ebe3e5edd/lm_eval/models/utils.py#L624
    """
    if tokenizer.pad_token:
        return tokenizer

    if tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


@dataclass
class EnsembleGCGConfig:
    num_steps: int = 250
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    topk: int = 256
    n_replace: int = 1
    batch_size: Optional[int] = None
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    allow_non_ascii: bool = False
    filter_ids: bool = True
    use_prefix_cache: bool = False  # kept False for portability across different Llama sizes
    early_stop_on_exact_prefix: bool = False  # if True, stop when exact prefix is matched
    add_space_before_target: bool = False
    seed: Optional[int] = 42
    verbosity: str = "INFO"

@dataclass
class EnsembleGCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]


def _sample_ids_from_grad(
    base_ids: Tensor,           # (L,)
    grad: Tensor,               # (L, V)
    search_width: int,
    topk: int,
    n_replace: int,
    not_allowed_ids: Optional[Tensor] = None,
) -> Tensor:
    L, V = grad.shape
    # Mask disallowed tokens by setting their gradient to +inf so they will not be chosen
    if not_allowed_ids is not None and not_allowed_ids.numel() > 0:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    # Choose topk per position by lowest gradient values (descending negative gradient)
    topk_ids = (-grad).topk(topk, dim=1).indices  # (L, topk)

    # For each candidate sequence, pick n_replace positions uniformly at random
    pos = torch.argsort(torch.rand((search_width, L), device=grad.device))[:, :n_replace]  # (W, n_replace)

    # For each chosen position, pick one id from that position's topk uniformly
    topk_choice = torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)  # (W, n_replace, 1)
    chosen_ids = torch.gather(topk_ids[pos], 2, topk_choice).squeeze(2)  # (W, n_replace)

    # Scatter into duplicates of base ids
    candidates = base_ids.unsqueeze(0).repeat(search_width, 1)  # (W, L)
    candidates.scatter_(1, pos, chosen_ids)
    return candidates  # (W, L)


class EnsembleGCG:
    def __init__(
        self,
        models: List[transformers.PreTrainedModel],
        tokenizers: List[transformers.PreTrainedTokenizer],
        config: EnsembleGCGConfig,
    ):
        assert len(models) >= 2, "Provide at least two models for transferability."
        assert len(models) == len(tokenizers), "One tokenizer per model."
        self.models = models
        self.toks = tokenizers
        self.cfg = config

        # Basic checks and setup
        # We assume all Llama 3.2 Instruct family share the same tokenizer merges/vocab.
        # We will use the first tokenizer for id space and decoding.
        self.main_tok = self.toks[0]

        # Some Llama tokenizers ship without pad token; set it to unk/eos if needed
        for t in self.toks:
            configure_pad_token(t)

        # Device handling: put all models on same device as the first
        self.device = next(self.models[0].parameters()).device
        for m in self.models:
            if next(m.parameters()).device != self.device:
                raise RuntimeError("All models must be on the same device.")

        # Not-allowed tokens: EOS/BOS/UNK/PAD plus any non-ascii unless allowed
        self.not_allowed = self._build_not_allowed()

    def _build_not_allowed(self) -> Tensor:
        ids = set()
        # common specials
        for special_id in [
            self.main_tok.bos_token_id,
            self.main_tok.eos_token_id,
            self.main_tok.unk_token_id,
            self.main_tok.pad_token_id,
        ]:
            if special_id is not None:
                ids.add(int(special_id))
        if not self.cfg.allow_non_ascii:
            for i in range(self.main_tok.vocab_size):
                s = self.main_tok.decode([i])
                # only printable ASCII tokens are allowed
                if not (s.isascii() and s.isprintable()):
                    ids.add(i)
        if len(ids) == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        return torch.tensor(sorted(list(ids)), dtype=torch.long, device=self.device)

    # ----- Core loss and gradient -----

    def _compute_ce_or_mellow(self, shift_logits: Tensor, shift_labels: Tensor) -> Tensor:
        # shift_logits: (B, T, V), shift_labels: (B, T)
        if self.cfg.use_mellowmax:
            picked = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            return mellowmax(-picked, alpha=self.cfg.mellowmax_alpha, dim=-1).mean(dim=-1)
        # default CE
        return torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.size(0), -1).mean(dim=-1)

    def _ids_to_embeds(self, model: transformers.PreTrainedModel, ids: Tensor) -> Tensor:
        # ids: (B, L)
        emb = model.get_input_embeddings()
        onehot = torch.nn.functional.one_hot(ids, num_classes=emb.num_embeddings).to(emb.weight.dtype)
        onehot = onehot.to(self.device)
        return onehot @ emb.weight  # (B, L, D)

    def _forward_loss_single_model(
        self,
        model: transformers.PreTrainedModel,
        before_ids: Tensor,  # (1, LB)
        cand_ids: Tensor,    # (B, Lopt)
        after_ids: Tensor,   # (1, LA)
        target_ids: Tensor,  # (1, LT)
    ) -> Tensor:
        # Build embeddings per model
        B = cand_ids.size(0)
        before_emb = self._ids_to_embeds(model, before_ids.repeat(B, 1))
        cand_emb = self._ids_to_embeds(model, cand_ids)
        after_emb = self._ids_to_embeds(model, after_ids.repeat(B, 1))
        tgt_emb = self._ids_to_embeds(model, target_ids.repeat(B, 1))
        inp = torch.cat([before_emb, cand_emb, after_emb, tgt_emb], dim=1)
        with torch.no_grad():
            out = model(inputs_embeds=inp)
            logits = out.logits
        # shift so token n-1 predicts token n in the target region
        LT = target_ids.shape[1]
        shift = inp.shape[1] - LT
        shift_logits = logits[:, shift - 1 : shift - 1 + LT, :]
        shift_labels = target_ids.repeat(B, 1)
        return self._compute_ce_or_mellow(shift_logits, shift_labels)  # (B,)

    def _forward_loss_ensemble(
        self,
        cand_ids: Tensor,
        before_ids_list: List[Tensor],
        after_ids_list: List[Tensor],
        target_ids_list: List[Tensor],
        search_batch_size: int,
    ) -> Tensor:
        # Evaluate each model in mini-batches, then average losses
        B = cand_ids.size(0)
        total_loss = torch.zeros((B,), device=self.device, dtype=torch.float32)
        for m_idx, model in enumerate(self.models):
            # batch over candidates to fit memory
            per_model_losses = []
            ids_before = before_ids_list[m_idx]
            ids_after = after_ids_list[m_idx]
            ids_target = target_ids_list[m_idx]
            for i in range(0, B, search_batch_size):
                j = min(B, i + search_batch_size)
                losses = self._forward_loss_single_model(
                    model,
                    ids_before,
                    cand_ids[i:j],
                    ids_after,
                    ids_target,
                )
                per_model_losses.append(losses)
            per_model_loss = torch.cat(per_model_losses, dim=0)  # (B,)
            total_loss += per_model_loss
        total_loss = total_loss / float(len(self.models))
        return total_loss  # (B,)

    def _token_grad_ensemble(
        self,
        optim_ids: Tensor,  # (1, Lopt)
        before_ids_list: List[Tensor],
        after_ids_list: List[Tensor],
        target_ids_list: List[Tensor],
    ) -> Tensor:
        # Build one-hot for main tokenizer space; compute gradient via each model's embeddings
        # using straight-through trick as in standard GCG. We average grads across models.
        L = optim_ids.shape[1]
        V = self.main_tok.vocab_size
        onehot = torch.nn.functional.one_hot(optim_ids, num_classes=V).to(self.device)
        onehot = onehot.to(self.models[0].get_input_embeddings().weight.dtype)
        onehot.requires_grad_()

        # Per-model forward with embeds to keep grad through onehot
        grads = []
        for m_idx, model in enumerate(self.models):
            emb = model.get_input_embeddings()
            before_ids = before_ids_list[m_idx]
            after_ids = after_ids_list[m_idx]
            target_ids = target_ids_list[m_idx]

            # Embeds with grad path to onehot
            opt_emb = torch.matmul(onehot, emb.weight)  # (1, Lopt, D)
            B = 1
            before_emb = self._ids_to_embeds(model, before_ids.repeat(B, 1))
            after_emb = self._ids_to_embeds(model, after_ids.repeat(B, 1))
            tgt_emb = self._ids_to_embeds(model, target_ids.repeat(B, 1))
            inp = torch.cat([before_emb, opt_emb, after_emb, tgt_emb], dim=1)

            out = model(inputs_embeds=inp)
            logits = out.logits

            LT = target_ids.shape[1]
            shift = inp.shape[1] - LT
            shift_logits = logits[:, shift - 1 : shift - 1 + LT, :]
            shift_labels = target_ids.repeat(B, 1)

            if self.cfg.use_mellowmax:
                picked = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                loss = mellowmax(-picked, alpha=self.cfg.mellowmax_alpha, dim=-1).mean()
            else:
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="mean",
                )

            grad = torch.autograd.grad(outputs=[loss], inputs=[onehot], retain_graph=True)[0]  # (1, L, V)
            grads.append(grad)

        avg_grad = torch.stack(grads, dim=0).mean(dim=0)  # (1, L, V)
        return avg_grad.squeeze(0)  # (L, V)

    # ----- Main optimization entrypoint -----

    def run(
        self,
        messages: Union[str, List[Dict[str, str]]],
        target_prefix: str,
        log_step: int = 20,
    ) -> EnsembleGCGResult:
        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        # Normalize messages format and template with {optim_str}
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if not any("{optim_str}" in m.get("content", "") for m in messages):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        # Build per-tokenizer chat templates and tokenize "static" parts
        # We use each tokenizer to prepare its own before/after/target ids.
        before_ids_list, after_ids_list, target_ids_list = [], [], []

        for tok in self.toks:
            template = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            bos = tok.bos_token or ""
            if bos and template.startswith(bos):
                template = template[len(bos):]
            before_str, after_str = template.split("{optim_str}")

            tgt_prefix = (" " + target_prefix) if self.cfg.add_space_before_target else target_prefix

            before_ids = tok([before_str], padding=False, return_tensors="pt")["input_ids"].to(self.device)
            after_ids = tok([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
            target_ids = tok([tgt_prefix], add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)

            before_ids_list.append(before_ids)
            after_ids_list.append(after_ids)
            target_ids_list.append(target_ids)

        # Initialize optimized tokens in the main tokenizer id space
        optim_ids = self.main_tok(self.cfg.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)

        best_loss = float("inf")
        best_string = self.main_tok.batch_decode(optim_ids)[0]
        losses, strings = [], []

        # Main loop
        for step in range(self.cfg.num_steps):
            # 1) Compute token gradient averaged across models
            avg_grad = self._token_grad_ensemble(optim_ids, before_ids_list, after_ids_list, target_ids_list)  # (L, V)

            # 2) Sample candidate ids using the gradient
            cand_ids = _sample_ids_from_grad(
                base_ids=optim_ids.squeeze(0),
                grad=avg_grad,
                search_width=self.cfg.search_width,
                topk=self.cfg.topk,
                n_replace=self.cfg.n_replace,
                not_allowed_ids=self.not_allowed,
            )  # (W, L)

            # 3) Evaluate ensemble loss on candidates (mini-batched)
            batch_size = self.cfg.batch_size or cand_ids.shape[0]
            compute = find_executable_batch_size(self._forward_loss_ensemble, batch_size)
            cand_losses = compute(
                cand_ids,
                before_ids_list,
                after_ids_list,
                target_ids_list,
            )  # (W,)

            # 4) Pick best and update
            idx = int(torch.argmin(cand_losses).item())
            cur_loss = float(cand_losses[idx].item())
            optim_ids = cand_ids[idx].unsqueeze(0)

            cur_string = self.main_tok.batch_decode(optim_ids)[0]
            losses.append(cur_loss)
            strings.append(cur_string)

            if cur_loss < best_loss:
                best_loss = cur_loss
                best_string = cur_string

            if (step + 1) % max(1, log_step) == 0:
                print(f"[EnsembleGCG] step={step+1} loss={cur_loss:.4f}")

        return EnsembleGCGResult(
            best_loss=best_loss,
            best_string=best_string,
            losses=losses,
            strings=strings,
        )
