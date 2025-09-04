from collections import defaultdict
import torch
import unicodedata
import numpy as np
import inspect

KEEP_COMBINING = set()  # e.g., set(["\u0308"]) to keep diaeresis

def normalize_greek(text: str, *, strip_diacritics: bool = True) -> str:
    if not isinstance(text, str):
        return text
    # 1) Decompose so diacritics become combining marks
    s = unicodedata.normalize("NFD", text)

    # 2) Unify variant apostrophes/koronis to a single mark (’)
    #    This helps reduce visual variants in Greek texts
    s = (
        s.replace("\u1FBD", "’")  # Greek Koronis
        .replace("\u02BC", "’")  # Modifier Letter Apostrophe
        .replace("\u2019", "’")  # Right single quotation mark
        .replace("\u00B4", "’")  # Spacing acute accent (rarely used)
        .replace("'", "’")       # ASCII apostrophe
    )

    # 3) Optionally strip all combining marks (accents, breathings, subscripts)
    if strip_diacritics:
        s = "".join(
            ch for ch in s
            if not (unicodedata.category(ch) == "Mn" and ch not in KEEP_COMBINING)
        )

    # 4) Recompose
    return unicodedata.normalize("NFC", s)


# 2) Helper: batched embeddings (last-4 layer average -> mean over tokens)
def bert_sentence_embeddings(
    texts: list[str],
    *,
    tokenizer,
    model,
    device="cpu",
    batch_size=32,
    max_length=None,
    l2_normalize=True,
):
    if max_length is None:
        max_length = tokenizer.model_max_length  # usually 512
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            # move tensors to device
            for k, v in enc.items():
                if isinstance(v, torch.Tensor):
                    enc[k] = v.to(device)

            out = model(**enc, output_hidden_states=True)
            # last four hidden layers -> average across layers
            last4 = out.hidden_states[-4:]                    # list of 4 tensors, each [B, T, H]
            token_reps = torch.stack(last4, dim=0).mean(0)    # [B, T, H]

            # mean-pool over tokens with attention mask (exclude padding)
            mask = enc["attention_mask"].unsqueeze(-1)        # [B, T, 1]
            masked = token_reps * mask
            lengths = mask.sum(dim=1).clamp(min=1)            # [B, 1]
            sent_reps = masked.sum(dim=1) / lengths           # [B, H]

            if l2_normalize:
                sent_reps = torch.nn.functional.normalize(sent_reps, p=2, dim=1)

            vecs.append(sent_reps.cpu().numpy())
    return np.vstack(vecs) if vecs else np.empty((0, model.config.hidden_size), dtype=np.float32)

# --------------------------------------------------------------
# Helper: Encode with safe truncation and proper device
# --------------------------------------------------------------
def encode_trunc(text: str, tokenizer, device="cpu", max_len=512):
    kwargs = {
        "text": text,
        "return_tensors": "pt",
        "truncation": True,
        "max_length": max_len,
    }
    # Preserve your check for optional arg
    sig = inspect.signature(tokenizer.__call__)
    if "add_special_tokens" in sig.parameters:
        kwargs["add_special_tokens"] = True

    enc = tokenizer(**kwargs)
    # Force-move every tensor to the right device
    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            enc[k] = v.to(device)
    return enc


# --------------------------------------------------------------
# Augmentation with subwords
# --------------------------------------------------------------
def augment_with_subwords(
    tokens: list[dict],
    *,
    tokenizer,
    anchor_use_lemma: bool,
    target_lemma: str
):
    target_lemma = target_lemma.lower()
    aug_tokens = []
    words = []
    sp_tokens = []
    sp_pos = 0

    prepend = tokenizer.cls_token or "<s>"
    append = tokenizer.sep_token or "</s>"
    if prepend:
        sp_tokens.append(prepend)
        sp_pos += 1

    for t in tokens:
        is_anchor = t["lemma"].lower() == target_lemma
        word = t["lemma"].lower() if is_anchor and anchor_use_lemma else t["token_text"].lower()

        try:
            word_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        except Exception:
            word_ids = tokenizer.encode(word, add_special_tokens=False)

        subwords = tokenizer.convert_ids_to_tokens(word_ids)

        new_t = dict(t)
        new_t["sp_first"] = sp_pos
        new_t["sp_pieces"] = subwords
        aug_tokens.append(new_t)

        words.append(word)
        sp_tokens.extend(subwords)
        sp_pos += len(subwords)

    if append:
        sp_tokens.append(append)

    sent_str = " ".join(words)
    return sent_str, sp_tokens, aug_tokens


# --------------------------------------------------------------
# Hidden-state embedding from model (average last 4 layers 8,9,10,11)
# --------------------------------------------------------------
def hidden_anchor_embedding(
    aug_tokens: list[dict],
    sent_str: str,
    *,
    tokenizer,
    model,
    device,
    target_lemma: str,
    layers: list[int] = (8, 9, 10, 11),  # layers to average
    piece_pooling: str = "mean",          # "mean" or "sum"
):
    target_lemma = target_lemma.lower()
    anchor = next((t for t in aug_tokens if t["lemma"].lower() == target_lemma), None)
    if anchor is None:
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    sp_first = anchor["sp_first"]
    k = len(anchor["sp_pieces"])

    enc = encode_trunc(
        sent_str,
        tokenizer=tokenizer,
        device=device,
        max_len=tokenizer.model_max_length,
    )

    with torch.no_grad():
        outs = model(**enc, output_hidden_states=True)
        # outs.hidden_states is a tuple of length num_layers+1 if embeddings are included, or num_layers otherwise.
        # We index the desired transformer layers and average them.
        try:
            per_layer = [outs.hidden_states[i].squeeze(0) for i in layers]  # each [seq_len, dim]
        except IndexError:
            # If a requested layer index is out of range, fall back to the last available 4 layers
            num_hs = len(outs.hidden_states)
            fallback = list(range(max(0, num_hs - 4), num_hs))
            per_layer = [outs.hidden_states[i].squeeze(0) for i in fallback]

        hidden = torch.stack(per_layer, dim=0).mean(dim=0)  # [seq_len, dim]

    if sp_first + k - 1 >= hidden.shape[0]:
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    span = hidden[sp_first : sp_first + k]  # [k, dim]
    vec = span.sum(dim=0) if piece_pooling == "sum" else span.mean(dim=0)
    return vec.detach().cpu().numpy()


# --------------------------------------------------------------
# Attention-based weighting of lemmas
# --------------------------------------------------------------
def attention_weights_by_lemma(
    aug_tokens: list[dict],
    sent_str: str,
    *,
    tokenizer,
    model,
    device,
    att_layer: int,
    target_lemma: str,
    top_k: int = 3,           # number of top attention heads to keep
    direction: str = "from",  # "from" = target attends to others, "to" = others attend to target
    normalize: bool = True,   # normalize anchor_vec to sum to 1
):
    target_lemma = target_lemma.lower()

    enc = encode_trunc(
        sent_str,
        tokenizer=tokenizer,
        device=device,
        max_len=tokenizer.model_max_length,
    )

    with torch.no_grad():
        outs = model(**enc, output_attentions=True)
        A_heads = outs.attentions[att_layer][0].cpu()  # [num_heads, L, L]

    anchor = next((t for t in aug_tokens if t["lemma"].lower() == target_lemma), None)
    if anchor is None:
        return {}

    sp_first = anchor["sp_first"]
    k_anchor = len(anchor["sp_pieces"])
    L = A_heads.shape[-1]

    if sp_first + k_anchor - 1 >= L:
        return {}

    if direction == "from":
        per_head_anchor_vecs = A_heads[:, sp_first : sp_first + k_anchor, :].mean(dim=1)
    elif direction == "to":
        per_head_anchor_vecs = A_heads[:, :, sp_first : sp_first + k_anchor].mean(dim=2)
    else:
        raise ValueError("direction must be 'from' or 'to'")

    head_scores = per_head_anchor_vecs.sum(dim=1)  # [H]
    top_head_ids = torch.topk(head_scores, k=top_k).indices
    anchor_vec = per_head_anchor_vecs[top_head_ids].mean(dim=0)  # [L]

    if normalize and anchor_vec.sum() > 0:
        anchor_vec = anchor_vec / anchor_vec.sum()

    lemma_info = defaultdict(lambda: {"weight": 0.0, "pieces": []})
    for t in aug_tokens:
        lemma = t["lemma"].lower()
        if lemma == target_lemma or lemma.strip() == "":
            continue

        start = t["sp_first"]
        end = start + len(t["sp_pieces"])
        if end > anchor_vec.shape[0]:
            continue

        total_w = anchor_vec[start:end].sum().item()
        lemma_info[lemma]["weight"] += total_w

        for j, piece in enumerate(t["sp_pieces"]):
            idx = start + j
            lemma_info[lemma]["pieces"].append({
                "piece": piece,
                "sp_idx": idx,
                "weight": float(anchor_vec[idx]),
            })

    return dict(lemma_info)