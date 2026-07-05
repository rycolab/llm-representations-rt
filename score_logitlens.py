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

def to_tokens_and_logprobs(model, tokenizer, input, mlen=2048):
    eps=1e-8
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
    print("Beginning to score batches...")
    input_ids = torch.cat((torch.tensor([[tokenizer.eos_token_id]]),
        tokenizer(input, padding="max_length", max_length=mlen, truncation=True, 
            return_tensors="pt").input_ids),1)
    print(input_ids)
    input_ids = input_ids.to(torch.device('cuda'))
    dataset = DatasetToScore(input_ids)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    logitlens_layers = {}
    tokens = []
    with torch.inference_mode():
        last_ln = model.transformer.ln_f
        lm_head = model.lm_head
        for batch in tqdm(train_loader):
            outputs = model(batch[:,:-1], output_hidden_states=True)
            reps = torch.stack(outputs[2])
            batch_ids = batch[:, 1:]
            target_ids = batch_ids[0]
            del outputs

            for layer_id, layer_rep in enumerate(reps):
                if layer_id not in logitlens_layers:
                    logitlens_layers[f"logitlens_{layer_id:02}"] = []
                if layer_id == len(reps) - 1:
                    layer_logit = lm_head(layer_rep)
                else:
                    layer_logit = lm_head(last_ln(layer_rep))
                surprisal_subwords = loss_fct(layer_logit.transpose(1,2), batch_ids) + EPS
                surprisal_subwords = surprisal_subwords[0].detach().cpu().float().tolist()
                for t_id, surp in zip(target_ids, surprisal_subwords):
                    if t_id not in tokenizer.all_special_ids:
                        if layer_id > 0:
                            logitlens_layers[f"logitlens_{layer_id:02}"].append(surp)
                        else:
                            tokens.append(tokenizer.decode(t_id))
                del surprisal_subwords
            del reps
    return tokens, logitlens_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input files. One document per file, one sentence per line.")
    parser.add_argument("--out", type=str, help="Path to output file where logit lens surprisals should be written.")
    parser.add_argument("--model", type=str, default="ai-forever/mGPT", help=f"Name of the model")
    args = parser.parse_args()

    input_files = sorted(os.listdir(args.input))

    print(f"Loading model {args.model}...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", config=config, trust_remote_code=True)
    model.config.pad_token_id = model.config.eos_token_id
    max_len = model.config.n_ctx

    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=max_len, padding_side="right", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    #model = model.to(torch.device('cuda'))
    print("Model fully loaded.")
    print(max_len)

    cols = "document_id token_id token".split()
    layer_names = [f"logitlens_{l:02}" for l in range(1, model.config.n_layer+1)]
    logitlens_all = {col:[] for col in cols + layer_names}
    for file in input_files:
        document_id = file.split(".txt")[0]
        text = read_file(args.input, file)
        document = text.replace('\n\n\n', ' ').replace('\n\n', ' ').replace('\n', ' ')
        tokens, logitlens_doc = to_tokens_and_logprobs(model, tokenizer, document, mlen=max_len)
        tok_ids = list(range(len(tokens)))
        logitlens_all["document_id"].extend([document_id for i in tok_ids])
        logitlens_all["token_id"].extend(tok_ids)
        logitlens_all["token"].extend(tokens)
        for layer_name in layer_names:
            logitlens_all[layer_name].extend(logitlens_doc[layer_name])
    df = pd.DataFrame(logitlens_all)
    df.to_csv(args.out, index=False)
