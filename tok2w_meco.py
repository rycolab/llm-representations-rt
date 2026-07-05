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


LANG_NAMES = 'English Greek Hebrew Russian Turkish'.split()
LANG_CODE = 'en gr he ru tr'.split()

LANG_MAP = dict(zip(LANG_NAMES, LANG_CODE))
LANG_IDXs = list(range(1, len(LANG_NAMES)+1))

measures = 'firstfix_rt gaze_rt total_rt'.split()
print_measures = 'First_Fixation_Duration Gaze_duration Total_Reading_Time'.split()
measure_dict = dict(zip(measures, print_measures))

def get_freq(word, lang_code, min_freq):
    # by frequency we mean unigram surprisal
    w_freq = word_frequency(word, lang_code)
    if w_freq == 0:
        w_freq = min_freq
    return -1 * np.log2(w_freq)

lang_help = f"Choose index between 1 and {len(LANG_NAMES)} for language of dataset: {', '.join(LANG_NAMES)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=str, help="Directory for surprisal dataframes")
    parser.add_argument("--durations", type=str, help="Directory for duration dataframes")
    parser.add_argument("--out", type=str, help="Directory for output files")
    parser.add_argument("--n_layers", type=int, default=24, help="number of embedding layers")
    parser.add_argument("--lang", type=int, default=1, help=lang_help, choices=LANG_IDXs)
    parser.add_argument("--lmodel", type=str, default='mgpt', help="gpt2 or mgpt predictor features", choices=['gpt2', 'mgpt'])
    args = parser.parse_args()

    start_time = time()

    scored_path = Path(args.scored)
    dur_path = Path(args.durations)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    n_layers = args.n_layers
    layers = list(range(1, n_layers+1))
    logit_names = [f"logitlens_{l:02}" for l in layers]
    logitlens_all = {col:[] for col in logit_names}

    lang_idx = args.lang
    lang = LANG_NAMES[lang_idx-1]
    lang_code = LANG_MAP[lang]

    surp = pd.read_csv(scored_path / "meco_surp" / f"{lang}.csv", keep_default_na=False).sort_values(by=['document_id', 'token_id']).reset_index(drop=True)
    logit_df = pd.read_csv(scored_path / "meco_logitlens" / f"{lang}.csv", keep_default_na=False).sort_values(by=['document_id', 'token_id']).reset_index(drop=True)
    surp = surp.merge(logit_df, on=['document_id', 'token_id', 'token'])

    iv = pd.read_csv(scored_path / "infoval" / f"meco_{lang_code}_k50_seed0_newtokens3_{args.lmodel}.csv", keep_default_na=False).sort_values(by=['document_id', 'word_id']).reset_index(drop=True)
    iv = iv[[f"infovalue_layer{l}" for l in layers]]

    # eye tracking aggregated
    dur_df = pd.read_csv(dur_path / f"{lang_code}.csv", keep_default_na=False).sort_values(by=['trialid', 'trialid']).reset_index(drop=True)

    if lang == "Turkish":
        surp = pd.concat([surp.iloc[:2413], surp.iloc[2414:]]).reset_index(drop=True)
        iv = pd.concat([iv.iloc[:1050], iv.iloc[1051:]]).reset_index(drop=True)
        for measure in measures:
            dur_df[measure].iloc[1194] = 0
        dur_df = pd.concat([dur_df.iloc[:1058], dur_df.iloc[1059:]]).reset_index(drop=True)

    emb_path = scored_path / "meco_emb" / lang

    out_emb_dir = out_path / "meco_embed_words" / lang
    out_emb_dir.mkdir(parents=True, exist_ok=True)

    for l in layers:
        ldir = out_emb_dir / f"layer_{l:02}"
        ldir.mkdir(parents=True, exist_ok=True)

    words = []
    word_numbers = []
    doc_ids = []

    prev_len = []
    prev_freq = []
    prevw_surprisals = []

    word_len = []
    freqs = []
    new_surprisals = []

    # language code of Greek is different in wordfreq from the duration dataframes
    lang_code = 'el' if lang_code=='gr' else lang_code
    freq_dict = get_frequency_dict(lang_code)
    min_freq = min([freq_dict[k] for k in freq_dict])

    doc_ids_unique = sorted(surp["document_id"].unique().tolist())
    for doc_id in doc_ids_unique:
        doc_df = dur_df[dur_df['trialid']==doc_id]
        doc_words = doc_df['ia'].tolist()
        doc_len = len(doc_words)
        words.extend(doc_words)
        word_numbers.extend(list(range(1, doc_len+1)))
        doc_ids.extend([doc_id for i in range(doc_len)])

        doc_freqs = [get_freq(w, lang_code, min_freq) for w in doc_words]

        freqs.extend(doc_freqs)
        w_lens = [len(w) for w in doc_words]
        word_len.extend(w_lens)

        surp_doc = surp[surp["document_id"]==doc_id]
        if lang == 'Turkish' and doc_id == 8:
            emb = {}
            for l in layers:
                l_emb = torch.load(emb_path / f"layer_{l:02}/Text_ID_{doc_id}.pt")
                emb[l] = torch.cat((l_emb[:62, :], l_emb[63:, :]))
        else:
            emb = {l: torch.load(emb_path / f"layer_{l:02}/Text_ID_{doc_id}.pt") for l in layers} # token embeddings

        tokens = surp_doc["token"].tolist()
        surp_vals = surp_doc["document_surprisal"].tolist()

        prev_len.append("NA")
        prev_freq.append("NA")
        prev_len.extend(w_lens[:-1])
        prev_freq.extend(doc_freqs[:-1])

        embeddings = {l: [] for l in layers} # output word embeddings

        current = 0 #token index
        prev_surp = "NA"
        for word in doc_words:
            word = word.replace('"', '')
            surprisal = 0
            tlen = 0
            wlen = len(word)
            start = current
            curr_tok = tokens[current].strip().replace('"', '')
            if not word.startswith(tokens[current].strip().replace('"', '')):
                print("Issue!!!", doc_id, current, word, tokens[current])
            for i in range(current, len(tokens)):
                tlen += len(tokens[i].strip().replace('"', ''))
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
            torch.save(layer_embeddings, out_emb_dir / f"layer_{l:02}/Text_ID_{doc_id:02}.pt")

    new_df = pd.DataFrame({
        "Text_ID": doc_ids,
        "Word_Number": word_numbers,
        "Word": words,
        "prev_len": prev_len,
        "prev_freq": prev_freq,
        "prev_surp": prevw_surprisals,
        "word_len": word_len,
        "freq": freqs,
        "document_surprisal": new_surprisals})

    for logit_name in logit_names:
        new_df[logit_name] = logitlens_all[logit_name]

    dur_df = dur_df.drop(dur_df.groupby(["trialid"]).head(1).index, axis=0).reset_index(drop=True)
    new_df = new_df.drop(new_df.groupby(["Text_ID"]).head(1).index, axis=0).reset_index(drop=True)
    for measure in measures:
        print_measure = measure_dict[measure]
        duration = dur_df[measure].tolist()
        new_df[print_measure] = duration
    new_df = new_df.join(iv)
    new_df.to_csv(out_path / f"{lang}.csv", index=False)
    print("Time:", timedelta(seconds=time()-start_time))
