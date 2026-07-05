import argparse
import string
import pandas as pd
import numpy as np
import torch
from wordfreq import word_frequency, get_frequency_dict

import random

from time import time
from datetime import timedelta
from pathlib import Path

measure_dict = {
    "IA_FIRST_FIXATION_DURATION": "First_Fixation_Duration",
    "IA_FIRST_RUN_DWELL_TIME": "Gaze_duration",
    "IA_DWELL_TIME": "Total_Reading_Time"
}
duration_measures = list(measure_dict.keys())

freq_dict = get_frequency_dict("en")
min_freq = min([freq_dict[k] for k in freq_dict])

def get_freq(word):
    # by frequency we mean unigram surprisal
    w_freq = word_frequency(word, 'en')
    if w_freq == 0:
        w_freq = min_freq
    return -1 * np.log2(w_freq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir", type=str, help="Directory for input and output files")
    parser.add_argument("--lmodel", type=str, default='mgpt', help="gpt2 or mgpt predictor features", choices=['gpt2', 'mgpt'])
    parser.add_argument("--n_layers", type=int, default=24, help="number of embedding layers")
    args = parser.parse_args()

    fpath = Path(args.fdir)

    n_layers = args.n_layers
    layers = list(range(1, n_layers+1))
    logit_names = [f"logitlens_{l:02}" for l in layers]
    logitlens_all = {col:[] for col in logit_names}

    et = pd.read_csv(fpath / "provo_et_clean.csv", keep_default_na=False)
    surp = pd.read_csv(fpath / "provo_scored.csv", keep_default_na=False)
    logit_df = pd.read_csv(fpath / "provo_logitlens.csv", keep_default_na=False)
    surp = surp.merge(logit_df, on=['document_id', 'token_id', 'token'])

    doc_col = "Text_ID"
    # word-level information value to merge later
    iv = pd.read_csv(fpath / "estimates" / "provo" / f"provo_k50_seed0_newtokens3_{args.lmodel}.csv", keep_default_na=False).sort_values(by=['document_id', 'word_id']).reset_index(drop=True)
    iv = iv[[f"infovalue_layer{l}" for l in layers]]
    iv.drop(iv.tail(2).index,inplace=True)

    out_emb_dir = fpath / "provo_embed_words"
    out_emb_dir.mkdir(parents=True, exist_ok=True)

    for l in layers:
        ldir = out_emb_dir / f"layer_{l:02}"
        ldir.mkdir(parents=True, exist_ok=True)

    doc_ids_unique = et[doc_col].unique().tolist()
    participants = et["Participant_ID"].unique().tolist()
    n_participants = len(participants)
    p1 = et[et["Participant_ID"]==participants[0]]

    words = []
    w_ids = []
    word_numbers = []
    doc_ids = []

    prev_len = []
    prev_freq = []
    prevw_surprisals = []

    word_len = []
    freqs = []
    new_surprisals = []

    emb_path = fpath / "provo_embed"

    for doc_id in doc_ids_unique:
        et_doc = p1[p1[doc_col]==doc_id]
        doc_words = [w for w in et_doc["Word"].tolist() if w!="NA"]
        words.extend(doc_words)
        w_ids.extend([w for w in et_doc["Word_Unique_ID"].tolist() if w!="NA"])
        word_numbers.extend([w for w in et_doc["Word_Number"].tolist() if w!="NA"])
        doc_ids.extend([doc_id for i in range(len(doc_words))])

        doc_freqs = [get_freq(w) for w in doc_words]
        freqs.extend(doc_freqs)

        w_lens = [len(w) for w in doc_words]
        word_len.extend(w_lens)

        surp_doc = surp[surp["document_id"]==doc_id]
        tokens = surp_doc["token"].tolist()
        surp_vals = surp_doc["document_surprisal"].tolist()
        emb = {l: torch.load(emb_path / f"layer_{l:02}/{doc_col}_{doc_id:02}.pt") for l in layers}

        # skip first word to match provo eye tracking data
        # but add first-word information to previous-word information columns
        current = None
        first_word_tokens = []
        prev_surp = 0
        for idx, tok in enumerate(tokens):
            if idx!=0 and tok.startswith(' '):
                current = idx
                break
            else:
                prev_surp += surp_vals[idx]
                first_word_tokens.append(tok.strip())

        first_word = ''.join(first_word_tokens)

        prev_freq.append(get_freq(first_word))
        prev_freq.extend(doc_freqs[:-1])

        prev_len.append(len(first_word))
        prev_len.extend(w_lens[:-1])

        embeddings = {l: [] for l in layers}

        for word in doc_words:
            surprisal = 0
            tlen = 0
            wlen = len(word)
            start = current
            if not word.startswith(tokens[current].strip()):
                print("Issue!!!", doc_id, current)
            for i in range(current, len(tokens)):
                tlen += len(tokens[i].strip())
                if tlen >= wlen:
                    current = i+1
                    surprisal = sum(surp_vals[start:current])
                    new_surprisals.append(surprisal)
                    prevw_surprisals.append(prev_surp)
                    prev_surp = surprisal
                    for l in layers:
                        embeddings[l].append(torch.mean(emb[l][start:current],dim=0))
                        logit_name = f"logitlens_{l:02}"
                        logitlens_all[logit_name].append(sum(surp_doc[logit_name].tolist()[start:current]))
                    break
        for l in layers:
            layer_embeddings = torch.stack(embeddings[l])
            torch.save(layer_embeddings, out_emb_dir / f"layer_{l:02}/{doc_col}_{doc_id:02}.pt")

    new_df = pd.DataFrame({
        doc_col: doc_ids,
        "Word_Unique_ID": w_ids,
        "Word_Number": word_numbers,
        "Word": words,
        "prev_len": prev_len,
        "prev_freq": prev_freq,
        "prev_surp": prevw_surprisals,
        "word_len": word_len,
        "freq": freqs,
        "document_surprisal": new_surprisals})

    for measure in duration_measures:
        print_measure = measure_dict[measure]
        skiplist = []
        measure_means = []
        for doc_id in doc_ids_unique:
            doc_df = et[et[doc_col]==doc_id]
            wnums = [w for w in doc_df["Word_Number"].unique().tolist() if w!="NA"]
            for wnum in wnums:
                w_df = doc_df[doc_df["Word_Number"]==wnum]
                if len(w_df)!=n_participants:
                    print("Problem!!", doc_id, wnum, len(w_df))
                    break
                measure_vals = w_df[measure]
                skiplist.append([0 if val in {'NA', '0'} else 1 for val in measure_vals])
                measure_vals = [float(m) for m in measure_vals if m!='NA']
                w_mean = np.array(measure_vals).mean() if len(measure_vals)>0 else 0
                measure_means.append(w_mean)
        new_df[print_measure] = measure_means

        skiplist = np.array(skiplist).transpose()
        if skiplist.shape[0] != n_participants:
            print("skiprates problem!!!")
        skip_dict = {doc_col: doc_ids}
        for participant, vec in zip(participants, skiplist):
            skip_dict[participant] = vec
        skip_df = pd.DataFrame(skip_dict)
        skip_df.to_csv(fpath / f"provo_skiprate_{print_measure}.csv", index=False)
    for logit_name in logit_names:
        new_df[logit_name] = logitlens_all[logit_name]
    new_df = new_df.join(iv)
    new_df.to_csv(fpath / "provo_word_surprisal.csv", index=False)
