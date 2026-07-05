import argparse
import random
import string
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

from copy import deepcopy
from datetime import timedelta
from sklearn import metrics, model_selection
from tabulate import tabulate
from time import time
from pathlib import Path


SEED = 42
np.random.seed(SEED)
random.seed(42)
N_SPLITS = 11 # we will exclude the validation set used in tuning
KF = model_selection.KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
decimals_rounding = 2

duration_measures = "First_Fixation_Duration Gaze_duration Total_Reading_Time".split()

alpha_candidates = [0.0, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 5.0, 10.0]
lt_wts = [1.0, 0.0] #Lasso vs Ridge

baseline_features = "prev_len prev_freq word_len freq".split()

def get_x(split_df, split_ids, doc_col, surp_col, baseline, featureset, emb_path, layer):
    if featureset in {"baseline", "embedding"}:
        x = split_df[baseline].to_numpy()
    elif featureset == "surprisal":
        x = split_df[baseline + [surp_col]].to_numpy()
    elif featureset == "logitlens":
        x = split_df[baseline + [f"logitlens_{layer:02}"]].to_numpy()
    elif featureset == "infoval":
        x = split_df[baseline + [f"infovalue_layer{layer}"]].to_numpy()
    if featureset == "embedding":
        embeddings = [torch.load(embed_path / f"{doc_col}_{split_id:02}.pt") for split_id in split_ids]
        embeddings = torch.cat(embeddings).detach().cpu().float().numpy()
        x = np.hstack((x, embeddings))
    x = sm.add_constant(x)
    return x



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Directory for input files")
    parser.add_argument("--out", type=str, help="Directory for output files")
    parser.add_argument("--features", type=str, default="baseline", help="Which feature setting to tune", choices=["baseline", "surprisal", "logitlens", "infoval", "embedding"])
    parser.add_argument("--layer", type=int, default=24, help="which embedding layer to tune", choices=list(range(1,25)))
    args = parser.parse_args()

    start = time()
    print("English")

    fpath = Path(args.input)
    layer = args.layer
    df = pd.read_csv(fpath / "provo_word_surprisal.csv", keep_default_na=False)
    embed_path = fpath / f"provo_embed_words/layer_{layer:02}"

    store_dict = {"measures": duration_measures}
    store_cols = "L1_trainMSE L1_valMSE L1_alpha L2_trainMSE L2_valMSE L2_alpha Regularize unreg_valMSE".split()
    for col in store_cols:
        store_dict[col] = []
    store_dir = Path(args.out) / "tuning"
    store_dir.mkdir(parents=True, exist_ok=True)

    feature_setting = args.features

    doc_col = 'Text_ID'
    doc_ids = df[doc_col].unique().tolist()
    surp_col = "document_surprisal"

    train_idx, val_idx = list(KF.split(doc_ids))[0]
    train_ids = sorted([doc_ids[i] for i in train_idx])
    train_df = df[df[doc_col].isin(train_ids)].copy()

    val_ids = sorted([doc_ids[i] for i in val_idx])
    print(f"Tuning validation set to exclude from experiments: {val_ids}\n")
    val_df = df[df[doc_col].isin(val_ids)].copy()

    train_X = get_x(train_df, train_ids, doc_col, surp_col, baseline_features, feature_setting, embed_path, layer)
    val_X = get_x(val_df, val_ids, doc_col, surp_col, baseline_features, feature_setting, embed_path, layer)

    for measure in duration_measures:
        print(f"Tuning for {measure}")
        train_Y = train_df[measure].to_numpy()
        val_Y = val_df[measure].to_numpy()
        model = sm.OLS(train_Y, train_X).fit()
        unreg_valMSE = round(metrics.mean_squared_error(val_Y, model.predict(val_X).tolist()), decimals_rounding)
        store_dict["unreg_valMSE"].append(unreg_valMSE)
        do_regularize = 0
        L1_best, L2_best = None, None
        for l1_wt in lt_wts:
            best_trainMSE, best_valMSE, best_alpha = 200000, 200000, None
            for alpha_w in alpha_candidates:
                model = sm.OLS(train_Y, train_X).fit_regularized(alpha=alpha_w, L1_wt=l1_wt)
                val_mse = metrics.mean_squared_error(val_Y, model.predict(val_X).tolist())
                if val_mse < best_valMSE:
                    best_trainMSE = metrics.mean_squared_error(train_Y, model.predict(train_X).tolist())
                    best_valMSE = val_mse
                    best_alpha = alpha_w
            best_trainMSE = round(best_trainMSE, decimals_rounding)
            best_valMSE = round(best_valMSE, decimals_rounding)
            if best_valMSE < unreg_valMSE:
                do_regularize = 1 # mark if we need regularization at all
            if l1_wt == 1.0:
                l_marker = 1
                L1_best = best_valMSE
            else:
                l_marker = 2
                L2_best = best_valMSE
            store_dict[f"L{l_marker}_trainMSE"].append(best_trainMSE)
            store_dict[f"L{l_marker}_valMSE"].append(best_valMSE)
            store_dict[f"L{l_marker}_alpha"].append(best_alpha)
        if do_regularize and L2_best < L1_best:
            do_regularize = 2
        store_dict["Regularize"].append(do_regularize)
    store_df = pd.DataFrame(store_dict)
    print(store_df)
    out_name = f"{feature_setting}_{layer:02}" if feature_setting in {"logitlens", "infoval", "embedding"} else feature_setting
    store_df.to_csv(store_dir / f"{out_name}.csv", index=False)
    print("Time:", timedelta(seconds=time()-start))
