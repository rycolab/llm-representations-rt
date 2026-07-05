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
    """Return the first word (stripped of punctuation) from a generated continuation."""
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


class IncrementalInfoValueScorer:
    """
    Maintains model state (KV cache) to efficiently compute embeddings
    and generate candidates as we iterate through a document.
    """
    def __init__(self, model, tokenizer, device, layer_indices):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer_indices = layer_indices
        self.past_key_values = None
        # Keep track of history IDs to use as input for generation
        self.history_ids = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)
        self.history_text_len = 0

    def reset(self):
        self.past_key_values = None
        self.history_ids = torch.tensor([], dtype=torch.long, device=self.device).unsqueeze(0)
        self.history_text_len = 0

    def update_history(self, word: str):
        """
        Updates the internal state with the new word (raw, unstripped).
        This advances the context for the next step.
        """
        # Prepend space if this is not the start of the document
        prefix = " " if self.history_text_len > 0 else ""
        text_chunk = prefix + word
        
        # Tokenize only the new chunk
        new_ids = self.tokenizer(text_chunk, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        
        # Run forward pass to update cache
        past_len = self.history_ids.shape[1]
        new_len = new_ids.shape[1]
        attention_mask = torch.ones((1, past_len + new_len), device=self.device)
        
        with torch.inference_mode():
            outputs = self.model(
                input_ids=new_ids,
                attention_mask=attention_mask,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
        
        self.past_key_values = outputs.past_key_values
        self.history_ids = torch.cat([self.history_ids, new_ids], dim=1)
        self.history_text_len += len(text_chunk)

    def generate_candidates(self, k: int, max_new_tokens: int) -> List[str]:
        """Generate k candidate next-word predictions."""
        if k <= 0:
            return []

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=self.history_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # outputs shape: [k, seq_len]
        # Remove the prompt tokens
        gen_texts: List[str] = []
        prompt_len = self.history_ids.shape[1]
        for seq in outputs:
            gen_ids = seq[prompt_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_texts.append(text)

        words: List[str] = []
        for t in gen_texts:
            w = first_generated_word(t)
            if w is not None and w != "":
                words.append(w)
        return words

    def score_words(self, words: List[str]) -> torch.Tensor:
        """
        Computes embeddings for a list of non-empty words given the CURRENT history.
        Returns tensor of shape [len(words), num_layers, hidden_dim].
        """
        if not words:
            return torch.empty(0, device=self.device)

        # Tokenize batch: prepend space to simulate continuation
        cand_texts = [" " + w for w in words]
        encoded = self.tokenizer(cand_texts, return_tensors="pt", padding=True, add_special_tokens=False)
        cand_ids = encoded["input_ids"].to(self.device)
        cand_mask = encoded["attention_mask"].to(self.device)
        
        batch_size = len(words)
        
        # Expand past_key_values to match batch size
        if self.past_key_values is None:
            expanded_past = None
            past_len = 0
        else:
            past_len = self.history_ids.shape[1]
            expanded_past = tuple(
                tuple(t.repeat(batch_size, 1, 1, 1) for t in layer_past)
                for layer_past in self.past_key_values
            )
            
        # Attention Mask: 1s for past, cand_mask for new tokens
        past_mask = torch.ones((batch_size, past_len), device=self.device)
        full_mask = torch.cat([past_mask, cand_mask], dim=1)
        
        with torch.inference_mode():
            outputs = self.model(
                input_ids=cand_ids,
                attention_mask=full_mask,
                past_key_values=expanded_past,
                output_hidden_states=True
            )
            
        # Extract embeddings
        all_layers = outputs.hidden_states
        selected_layers = [all_layers[li] for li in self.layer_indices]
        stack = torch.stack(selected_layers, dim=0) # [L, B, S, D]
        
        # Vectorized mean pooling with masking
        # cand_mask: [B, S] -> [1, B, S, 1]
        mask_expanded = cand_mask.unsqueeze(0).unsqueeze(-1).to(stack.dtype)
        
        # Sum embeddings over sequence dimension (S) where mask is 1
        sum_embeddings = (stack * mask_expanded).sum(dim=2) # [L, B, D]
        
        # Count valid tokens per sequence
        lengths = mask_expanded.sum(dim=2) # [1, B, 1]
        lengths = lengths.clamp(min=1.0) # Avoid division by zero
        
        # Compute mean
        mean_embeddings = sum_embeddings / lengths # [L, B, D]
        
        # Permute to [B, L, D] to match expected output
        return mean_embeddings.permute(1, 0, 2)


def main():
    parser = argparse.ArgumentParser(description="Score information value per word using mGPT (Optimized).")
    parser.add_argument("--input", type=str, required=True, help="Path to input files.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file.")
    parser.add_argument("--model_name", type=str, default="ai-forever/mGPT", help="HF model name.")
    parser.add_argument("--k", type=int, default=5, help="Number of candidate sequences.")
    parser.add_argument("--max_new_tokens", type=int, default=3, help="Max new tokens to generate.")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto | cpu | cuda.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--layers", type=str, default="all", help="Layers to use: 'all' or '0,11,23'.")
    parser.add_argument("--split_symbol", type=str, default=None, help="Symbol to split sentences.")
    parser.add_argument("--keep_punctuation", action="store_true", help="Keep punctuation on words.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    input_root = Path(args.input)
    output_path = Path(args.output)

    # Set random seeds
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

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()

    # Determine layers
    num_layers_model = getattr(model.config, "n_layer", None) or getattr(model.config, "num_hidden_layers", None)
    if num_layers_model is None:
        dummy_ids = tokenizer("", return_tensors="pt")[["input_ids"]].to(device)
        with torch.no_grad():
            dummy_out = model(dummy_ids, output_hidden_states=True)
        num_layers_model = len(dummy_out.hidden_states)

    if args.layers.lower() == "all":
        layer_indices = list(range(num_layers_model + 1))
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
        layer_indices = sorted(set(parsed)) or list(range(num_layers_model))

    logging.info(f"Model: {args.model_name}")
    logging.info(f"Using device: {device}")
    logging.info(f"Total model layers: {num_layers_model}")
    logging.info(f"Using layers: {layer_indices}")

    information_value = []
    strip_punctuation = not args.keep_punctuation

    if input_root.is_dir():
        input_files = sorted(p for p in input_root.iterdir() if p.is_file())
    else:
        input_files = [input_root]

    logging.info(f"Found {len(input_files)} document file(s) to process.")

    scorer = IncrementalInfoValueScorer(model, tokenizer, device, layer_indices)

    for file_path in tqdm(input_files, desc="Documents"):
        stem = file_path.stem
        if stem.isdigit():
            document_id = int(stem)
        else:
            document_id = stem

        logging.info(f"Processing file {file_path} (document_id={document_id})")

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
            continue

        scorer.reset()
        
        # Initialize context with the first word
        scorer.update_history(doc_words[0])

        # Iterate starting from the second word
        for i in range(1, len(doc_words)):
            if i % 10 == 0:
                logging.info(f"Processing word {i}/{len(doc_words)} in document {document_id}")

            target_word_raw = doc_words[i]
            target_word_clean = clean_word(target_word_raw, strip_punctuation=strip_punctuation)
            
            if not target_word_clean:
                scorer.update_history(target_word_raw)
                continue

            # 1. Generate candidates
            candidates = scorer.generate_candidates(k=args.k, max_new_tokens=args.max_new_tokens)
            
            # 2. Prepare list of words to score
            cands_clean = [clean_word(c, strip_punctuation=strip_punctuation) for c in candidates]
            cands_clean = [c for c in cands_clean if c]
            
            if not cands_clean:
                scorer.update_history(target_word_raw)
                continue

            words_to_score = [target_word_clean] + cands_clean
            
            # 3. Compute embeddings in batch
            embeddings = scorer.score_words(words_to_score)
            
            target_emb = embeddings[0]
            cand_embs = embeddings[1:]
            
            # 4. Calculate Info Value
            cand_stack = cand_embs.unsqueeze(1) # [num_cand, 1, num_layers, hidden_dim] -> wait, shape is [num_cand, L, D]
            # embeddings shape: [1+num_cand, L, D]
            # target_emb: [L, D]
            # cand_embs: [num_cand, L, D]
            
            # We need to compute cosine sim for each layer
            # target_emb: [L, D]
            # cand_embs: [C, L, D]
            
            # Transpose cand_embs to [L, C, D] to iterate over layers easily or use vectorized op
            cand_stack = cand_embs.transpose(0, 1) # [L, C, D]
            
            entry = {
                "document_id": document_id,
                "word_id": i,
                "word": target_word_clean,
            }

            for logical_idx in range(len(layer_indices)):
                layer_id = layer_indices[logical_idx]
                t_vec = target_emb[logical_idx]  # [D]
                c_vecs = cand_stack[logical_idx] # [C, D]
                
                # Cosine distance = 1 - cosine similarity
                cos_sim = F.cosine_similarity(c_vecs, t_vec.unsqueeze(0), dim=-1)
                cos_dist = 1.0 - cos_sim
                cos_dist = torch.nan_to_num(cos_dist, nan=0.0)
                
                entry[f"infovalue_layer{layer_id}"] = float(cos_dist.mean().item())

            information_value.append(entry)
            
            # 5. Update history
            scorer.update_history(target_word_raw)

    df = pd.DataFrame(information_value)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
