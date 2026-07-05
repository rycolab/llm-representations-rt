import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import os
import math
from pathlib import Path

EPS = 1e-8


class DatasetToScore(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return self.encodings[idx]

    def __len__(self):
        return len(self.encodings)

def read_file(fdir, file):
    text = ''
    with open(os.path.join(fdir, file), 'r') as f:
        text = f.read()
    return text

def to_tokens_and_logprobs(model, tokenizer, input, loss_fct, last_ln, lm_head, mlen=2048):
    print("Beginning to score batches...")
    input_ids = torch.cat((torch.tensor([[tokenizer.eos_token_id]]),
                           tokenizer(input, padding="max_length", max_length=mlen, truncation=True, return_tensors="pt").input_ids),1)[:,:-1]
    print(input_ids)
    input_ids = input_ids.to(torch.device('cuda'))
    dataset = DatasetToScore(input_ids)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    to_return = []
    embeddings = {}
    logitlens_surprisal = {}
    with torch.inference_mode():
        for batch in tqdm(train_loader):
            outputs = model(batch, use_cache=False, output_hidden_states=True)
            probs = torch.softmax(outputs.logits, dim=-1).detach()
            surprisals = -1 * torch.log2(probs)
            hstates = torch.stack(outputs.hidden_states)[1:, :, :-1, :] #exclude first layer of positional embeddings
            len_hstates = len(hstates)

            # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
            surprisals = surprisals[:, :-1, :]
            input_ids = batch[:, 1:]
            gen_surprisals = torch.gather(surprisals, 2, input_ids[:, :, None]).squeeze(-1)
            # gather all the surprisals for the sequences into a neat table
            for document, input_surprisals in zip(input_ids, gen_surprisals):
                for token, p in zip(document, input_surprisals):
                    if token not in tokenizer.all_special_ids:
                        to_return.append({
                            "token": tokenizer.decode(token),
                            "surprisal": p.item()
                        })
                for layer_n, layer_emb in enumerate(hstates):
                    if layer_n == len_hstates-1:
                        layer_logit = lm_head(layer_emb)
                    else:
                        layer_logit = lm_head(last_ln(layer_emb))
                    logitlens_surp = loss_fct(layer_logit.transpose(1,2), input_ids) + EPS
                    logitlens_surp = logitlens_surp[0].tolist()

                    layer_embeddings = layer_embeddings[0]
                    if layer_n not in embeddings:
                        embeddings[layer_n] = []
                        logitlens_surp[f"logitlens_{layer_n:02}"] = []
                    for token, embedding, logit_tok in zip(document, layer_embeddings, logitlens_surp):
                        if token not in tokenizer.all_special_ids:
                            embeddings[layer_n].append(embedding)
                            logitlens_surp[f"logitlens_{layer_n:02}"].append(logit_tok)
    embeddings = {layer_n: torch.stack(embeddings[layer_n]) for layer_n in embeddings}
    return to_return, embeddings, logitlens_surprisal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input files. One document per file, one sentence per line.")
    parser.add_argument("--out_surp", type=str, help="Path to output file where surprisals should be written.")
    parser.add_argument("--out_emb", type=str, help="Path to output directory where embeddings should be stored.")
    parser.add_argument("--model", type=str, default="ai-forever/mGPT", help=f"Name of the model")
    args = parser.parse_args()

    input_files = sorted(os.listdir(args.input))

    out_emb = Path(args.out_emb)
    out_emb.mkdir(parents=True, exist_ok=True)
    max_len = 2048
    if args.model == 'openai-community/gpt':
        max_len = 1024

    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=max_len, padding_side="right", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model}...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", config=config, trust_remote_code=True)
    model.config.pad_token_id = model.config.eos_token_id
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
    last_ln = model.transformer.ln_f
    lm_head = model.lm_head
    #model = model.to(torch.device('cuda'))
    print("Model fully loaded.")

    surprisals = []
    for file in input_files:
        document_id = file.split(".txt")[0]
        text = read_file(args.input, file)
        document = text.replace('\n\n\n', ' ').replace('\n\n', ' ').replace('\n', ' ')
        document_surprisals, embeddings, logitlens = to_tokens_and_logprobs(model, tokenizer, document, loss_fct, last_ln, lm_head, mlen=max_len)

        for i, item in enumerate(document_surprisals):
            surprisals.append({
                "document_id": document_id,
                "token_id": i,
                "token": item["token"],
                "document_surprisal": item["surprisal"]
            })
        for key in logitlens:
            surprisals[key] = logitlens[key]
        print(f"Saving document {document_id}")
        for layer_n in embeddings:
            layer_embeddings = embeddings[layer_n]
            layer_dir = out_emb / f"layer_{layer_n:02}"
            if not layer_dir.exists():
                layer_dir.mkdir(parents=True)
            torch.save(layer_embeddings, layer_dir / f"Text_ID_{document_id}.pt")
    df = pd.DataFrame(surprisals)
    df.to_csv(args.out_surp, index=False)
