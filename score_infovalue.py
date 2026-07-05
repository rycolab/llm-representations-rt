import argparse
import logging
import random
import string
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Punctuation for word boundary detection; excludes apostrophe to keep contractions intact
PUNCTUATION_FOR_WORD_BOUNDARY = string.punctuation.replace("'", "") + "“”‘’«»—–…"


def clean_word(word: str, strip_punctuation: bool = True) -> str:
    """Optionally strip leading/trailing punctuation from a word token."""
    word = word.strip()
    if not strip_punctuation:
        return word
    return word.strip(PUNCTUATION_FOR_WORD_BOUNDARY)


def first_generated_word(generated_text: str) -> Optional[str]:
    """Return the first word (stripped of punctuation) from a generated continuation.

    A word ends when we hit whitespace or punctuation. Leading whitespace is skipped.
    """
    if not generated_text:
        return None

    # Skip leading whitespace
    i = 0
    n = len(generated_text)
    while i < n and generated_text[i].isspace():
        i += 1

    if i >= n:
        return None

    chars = []
    while i < n:
        ch = generated_text[i]
        if ch.isspace() or ch in PUNCTUATION_FOR_WORD_BOUNDARY:
            break
        chars.append(ch)
        i += 1

    word = "".join(chars).strip(PUNCTUATION_FOR_WORD_BOUNDARY)
    return word or None


def generate_candidate_words(
    model,
    tokenizer,
    context_text: str,
    k: int,
    max_new_tokens: int,
    device: torch.device,
) -> List[str]:
    """Generate k candidate next-word predictions (as word strings).

    Uses autoregressive generation up to `max_new_tokens` and then cuts
    each continuation at the end of the first word.
    """
    if k <= 0:
        return []

    if context_text:
        inputs = tokenizer(context_text, return_tensors="pt")
    else:
        inputs = tokenizer("", return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # outputs shape: [k, seq_len]
    # Remove the prompt tokens to get only the generated continuation
    gen_texts: List[str] = []
    prompt_len = input_ids.shape[1]
    for seq in outputs:
        gen_ids = seq[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        gen_texts.append(text)

    words: List[str] = []
    for t in gen_texts:
        w = first_generated_word(t)
        if w is not None and w != "":
            words.append(w)

    return words


def get_word_embedding_layers(
    model,
    tokenizer,
    context_text: str,
    word: str,
    device: torch.device,
    layer_indices: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    """Return contextualised embeddings for `word` given `context_text`.

    Returns a tensor of shape [num_layers, hidden_dim], where each row
    is the mean over all tokens corresponding to `word` in that context.
    """
    word = word.strip()
    if not word:
        return None

    # Tokenize context and context+word with identical settings so we
    # can infer the token span for the word as the suffix.
    context_inputs = tokenizer(
        context_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    full_text = (context_text + (" " if context_text else "") + word)
    full_inputs = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    context_ids = context_inputs["input_ids"].to(device)
    full_ids = full_inputs["input_ids"].to(device)

    context_len = context_ids.shape[1]
    total_len = full_ids.shape[1]
    if total_len <= context_len:
        return None

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states  # tuple length num_layers+1
    if hidden_states is None or len(hidden_states) == 0:
        return None

    # Include all layers including embedding layer at index 0
    all_layers = hidden_states
    if layer_indices is None:
        layer_indices = list(range(len(all_layers)))

    layer_tensors = []
    token_slice = slice(context_len, total_len)
    for li in layer_indices:
        if li < 0 or li >= len(all_layers):
            raise ValueError(f"Invalid layer index: {li}")
        layer_h = all_layers[li]
        # layer_h: [1, seq_len, hidden_dim]
        word_tokens = layer_h[0, token_slice, :]
        if word_tokens.numel() == 0:
            return None
        layer_vec = word_tokens.mean(dim=0)  # [hidden_dim]
        layer_tensors.append(layer_vec)

    return torch.stack(layer_tensors, dim=0)  # [num_layers, hidden_dim]


def compute_infovalue_for_word(
    model,
    tokenizer,
    context_text: str,
    target_word: str,
    k: int,
    max_new_tokens: int,
    device: torch.device,
    layer_indices: Optional[List[int]] = None,
    strip_punctuation: bool = True,
) -> Optional[List[float]]:
    """Compute information value (avg cosine distance) per layer for one word.

    Returns a list of floats, one per layer, or None if computation fails.
    """
    target_word = clean_word(target_word, strip_punctuation=strip_punctuation)
    if not target_word:
        return None

    # Generate k candidate prediction words
    candidates = generate_candidate_words(
        model=model,
        tokenizer=tokenizer,
        context_text=context_text,
        k=k,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    if not candidates:
        return None

    # Ground-truth embedding
    target_emb = get_word_embedding_layers(
        model=model,
        tokenizer=tokenizer,
        context_text=context_text,
        word=target_word,
        device=device,
        layer_indices=layer_indices,
    )
    if target_emb is None:
        return None

    cand_embs = []
    for cand in candidates:
        cand_clean = clean_word(cand, strip_punctuation=strip_punctuation)
        if not cand_clean:
            continue
        emb = get_word_embedding_layers(
            model=model,
            tokenizer=tokenizer,
            context_text=context_text,
            word=cand_clean,
            device=device,
            layer_indices=layer_indices,
        )
        if emb is not None:
            cand_embs.append(emb)

    if not cand_embs:
        return None

    # Stack candidate embeddings: [num_cand, num_layers, hidden_dim]
    cand_stack = torch.stack(cand_embs, dim=0)
    num_layers, hidden_dim = target_emb.shape

    infovalues: List[float] = []
    for layer_idx in range(num_layers):
        t_vec = target_emb[layer_idx]  # [hidden_dim]
        c_vecs = cand_stack[:, layer_idx, :]  # [num_cand, hidden_dim]
        # Cosine distance = 1 - cosine similarity
        cos_sim = F.cosine_similarity(c_vecs, t_vec.unsqueeze(0), dim=-1)
        cos_dist = 1.0 - cos_sim
        # Handle potential NaNs
        cos_dist = torch.nan_to_num(cos_dist, nan=0.0)
        infovalues.append(float(cos_dist.mean().item()))

    return infovalues


def main():
    parser = argparse.ArgumentParser(description="Score information value per word using mGPT.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input files. One document per file, one sentence per line.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for information values.")
    parser.add_argument("--model_name", type=str, default="ai-forever/mGPT", help="HF model name (default: ai-forever/mGPT).")
    parser.add_argument("--k", type=int, default=5, help="Number of candidate sequences per timestep.")
    parser.add_argument("--max_new_tokens", type=int, default=3, help="Max new tokens to generate for candidates.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto | cpu | cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0).")
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to use: 'all' (default) or comma-separated 0-based indices, e.g. '0,11,23'.",
    )
    parser.add_argument(
        "--split_symbol",
        type=str,
        default=None,
        help="Optional symbol to split sentences on when collecting words (default: any whitespace).",
    )
    parser.add_argument(
        "--keep_punctuation",
        action="store_true",
        help="Keep leading/trailing punctuation on words (default: strip).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    input_root = Path(args.input)
    output_path = Path(args.output)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Some GPT-style models have no pad token; use EOS as pad in that case.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    # Determine number of transformer layers and which ones to use
    num_layers_model = getattr(model.config, "n_layer", None) or getattr(model.config, "num_hidden_layers", None)
    if num_layers_model is None:
        # Fallback: run a tiny forward pass to infer from hidden states
        dummy_ids = tokenizer("", return_tensors="pt")[["input_ids"]].to(device)
        with torch.no_grad():
            dummy_out = model(dummy_ids, output_hidden_states=True)
        num_layers_model = len(dummy_out.hidden_states)

    if args.layers.lower() == "all":
        layer_indices = list(range(num_layers_model + 1))  # Include embedding layer
    else:
        parts = [p.strip() for p in args.layers.split(",") if p.strip()]
        parsed = []
        for p in parts:
            try:
                idx = int(p)
                if 0 <= idx < num_layers_model:
                    parsed.append(idx)
            except ValueError:
                continue
        # Deduplicate and sort
        layer_indices = sorted(set(parsed)) or list(range(num_layers_model))

    logging.info(f"Model: {args.model_name}")
    logging.info(f"Using device: {device}")
    logging.info(f"Total model layers: {num_layers_model}")
    logging.info(f"Using layers: {layer_indices}")

    information_value = []

    # Whether to strip punctuation from words downstream
    strip_punctuation = not args.keep_punctuation

    # Determine list of input files: directory of files or single file.
    if input_root.is_dir():
        input_files = sorted(p for p in input_root.iterdir() if p.is_file())
    else:
        input_files = [input_root]

    logging.info(f"Found {len(input_files)} document file(s) to process.")

    for file_path in tqdm(input_files, desc="Documents"):
        # Derive document_id from filename (e.g., "01.txt" -> 1 or "01")
        stem = file_path.stem
        if stem.isdigit():
            document_id: Optional[int] | str = int(stem)
        else:
            document_id = stem

        logging.info(f"Processing file {file_path} (document_id={document_id})")

        # Collect all words in the document (across sentences/lines)
        doc_words: List[str] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                sentence = line.strip()
                if not sentence:
                    continue
                if args.split_symbol:
                    doc_words.extend(sentence.split(args.split_symbol))
                else:
                    doc_words.extend(sentence.split())

        if len(doc_words) <= 1:
            # Not enough words to form a prediction (we always skip the first word)
            continue

        # Always skip the first word in the document and use it as initial context.
        # For each position i (starting at 1), context consists of all previous words
        # doc_words[:i], and the target word is doc_words[i].
        for i in range(1, len(doc_words)):
            context_words = doc_words[:i]
            context_text = " ".join(context_words)
            word = doc_words[i]

            infovalues = compute_infovalue_for_word(
                model=model,
                tokenizer=tokenizer,
                context_text=context_text,
                target_word=word,
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                device=device,
                layer_indices=layer_indices,
                strip_punctuation=strip_punctuation,
            )

            if infovalues is None:
                continue

            entry = {
                "document_id": document_id,
                "word_id": i,
                "word": clean_word(word, strip_punctuation=strip_punctuation),
            }

            for logical_idx, value in enumerate(infovalues):
                layer_id = layer_indices[logical_idx]
                entry[f"infovalue_layer{layer_id}"] = value

            information_value.append(entry)

    # Convert to DataFrame and save as CSV once after processing all documents
    df = pd.DataFrame(information_value)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
