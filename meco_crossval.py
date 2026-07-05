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
N_SPLITS = 10
KF = model_selection.KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
decimals_rounding = 5

LANG_NAMES = 'English Greek Hebrew Russian Turkish'.split()
LANG_IDXs = list(range(1, len(LANG_NAMES)+1))
lang_help = f"Choose index between 1 and {len(LANG_NAMES)} for language of dataset: {', '.join(LANG_NAMES)}"

duration_measures = 'First_Fixation_Duration Gaze_duration Total_Reading_Time'.split()
baseline_features = "prev_len prev_freq word_len freq".split()
tune_val_ids = [10, 11]


def get_hyperparameters(tune_df, measure):
    df = tune_df[tune_df["measures"]==measure]
    do_reg = df["Regularize"].iloc[0]
    if do_reg == 0:
        l1_wt, alpha_w = 0, 0
    else:
        l1_wt = 1.0 if do_reg == 1 else 0.0 # Lasso or Ridge
        alpha_w = df["L1_alpha"].iloc[0] if do_reg == 1 else df["L2_alpha"].iloc[0]
    return do_reg, l1_wt, alpha_w


def get_xy(df, doc_ids, split_idx, cols, baseline, featureset, emb_path, layer=None):
    doc_col, surp_col, duration_col = cols
    split_ids = [doc_ids[i] for i in split_idx]
    split_df = df[df[doc_col].isin(split_ids)].copy()
    locs = split_df.index.to_list()
    features = deepcopy(baseline)
    if featureset == "surprisal":
        features += [surp_col]
    elif featureset == "logitlens":
        features += [f"logitlens_{layer:02}"]
    elif featureset == "infoval":
        features += [f"infovalue_layer{layer}"]
    x = split_df[features].to_numpy()
    if featureset == "embedding":
        # drop embedding that corresponds to first word
        embeddings = [torch.load(embed_path / f"{doc_col}_{split_id:02}.pt")[1:, :] for split_id in split_ids]
        embeddings = torch.cat(embeddings).detach().cpu().float().numpy()
        x = np.hstack((x, embeddings))
    x = sm.add_constant(x)
    y = split_df[duration_col].to_numpy()
    return x, y, locs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Directory for input files")
    parser.add_argument("--out", type=str, help="Directory for output files")
    parser.add_argument("--lang", type=int, default=1, help=lang_help, choices=LANG_IDXs)
    parser.add_argument("--features", type=str, help="Which feature setting to train on", choices=["baseline", "surprisal", "logitlens", "infoval", "embedding"])
    parser.add_argument("--layer", type=int, default=24, help="which embedding layer to train on", choices=list(range(1,25)))
    parser.add_argument("--permute", type=int, default=0, help="1 if you want to permute the data, 0 if not", choices=[0,1])

    args = parser.parse_args()
    start = time()

    in_dir = args.input
    store_dir = args.out
    feature_setting = args.features
    layer = args.layer
    do_permute = True if args.permute==1 else False
    lang_idx = args.lang
    lang = LANG_NAMES[lang_idx-1]
    print(lang)

    base_path = Path(args.input)
    fpath = base_path / "meco_w"
    full_df = pd.read_csv(fpath / f"{lang}.csv")
    embed_path = fpath / "meco_embed_words" / lang / f"layer_{layer:02}"

    if feature_setting in {"baseline", "surprisal"}:
        tune_name = feature_setting
    elif feature_setting in {"logitlens", "infoval", "embedding"}:
        tune_name = f"{feature_setting}_{layer:02}"
    tuned_path = base_path / "tuning" / lang
    # uncomment to run with Provo's tuning hyperparameters
    # if lang == "English":
    #     tuned_path = base_path / "tuning"
    tune_df = pd.read_csv(tuned_path / f"{tune_name}.csv", keep_default_na=False)


    doc_col = 'Text_ID'
    all_doc_ids = full_df[doc_col].unique().tolist()
    # remove tuning set
    doc_ids = [doc_id for doc_id in all_doc_ids if doc_id not in tune_val_ids]
    df = full_df[full_df[doc_col].isin(doc_ids)].copy().reset_index(drop=True)
    df_len = len(df)

    surp_col = "document_surprisal"

    to_add = "_permuted" if do_permute else ""
    store_path = Path(f"{store_dir}{to_add}") / lang
    models_path_base = store_path / f"models{to_add}"
    out_path_base = store_path / f"outputs{to_add}"

    for duration_col in duration_measures:
        setting_name = f"{feature_setting}_layer_{layer:02}" if feature_setting in {"logitlens", "infoval", "embedding"} else feature_setting

        models_path = models_path_base / duration_col / setting_name
        models_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path_base  / duration_col / setting_name
        out_path.mkdir(parents=True, exist_ok=True)

        do_regularize, l1_wt, alpha_w = get_hyperparameters(tune_df, duration_col)

        cols = doc_col, surp_col, duration_col
        fold_count = 1

        scores = {"fold": [], "val_mse": [], "train_mse": [], "log_lik": []}

        preds_dict = {
            doc_col: df[doc_col],
            duration_col: df[duration_col],
            "fold": [None] * df_len,
            setting_name: [None] * df_len
        }

        for train_idx, val_idx in KF.split(doc_ids):
            print(f"Fold {fold_count}")
            train_X, train_Y, _ = get_xy(df, doc_ids, train_idx, cols, baseline_features, feature_setting, embed_path, layer)
            val_X, val_Y, val_locs = get_xy(df, doc_ids, val_idx, cols, baseline_features, feature_setting, embed_path, layer)

            if do_permute:
                train_Y = np.random.permutation(train_Y)

            if do_regularize == 0:
                model = sm.OLS(train_Y, train_X).fit()
                log_lik = model.llf
            else:
                model_ols = sm.OLS(train_Y, train_X)
                model = model_ols.fit_regularized(alpha=alpha_w, L1_wt=l1_wt)
                model_refit = model_ols.fit(params=model.params) # refit parameters might be skewed
                log_lik = model_refit.llf

            train_Yhat = model.predict(train_X).tolist()
            train_mse = metrics.mean_squared_error(train_Y, train_Yhat)

            val_Yhat = model.predict(val_X).tolist()
            val_mse = metrics.mean_squared_error(val_Y, val_Yhat)

            for j, vloc in enumerate(val_locs):
                preds_dict[setting_name][vloc] = val_Yhat[j]
                preds_dict["fold"][vloc] = fold_count

            if do_regularize > 0:
                model = model_refit

            model.save(models_path / f"fold_{fold_count:02}.pkl", remove_data=True)

            scores["fold"].append(fold_count)
            scores["val_mse"].append(val_mse)
            scores["train_mse"].append(train_mse)
            scores["log_lik"].append(log_lik)
            fold_count +=1

        true_values = np.array(preds_dict[duration_col])
        predicted_values = np.array(preds_dict[setting_name])
        preds_dict["sq_error"] = (true_values - predicted_values)**2

        scores["fold"].append("std")
        scores["fold"].append("mean")
        scores["fold"].append("dataset")
        for score in "val_mse train_mse log_lik".split():
            scores[score].append(np.std(scores[score]))
            scores[score].append(np.average(scores[score]))
            if score == "val_mse":
                scores[score].append(metrics.mean_squared_error(true_values, predicted_values))
            else:
                scores[score].append(np.nan)
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(out_path / "metrics.csv", index=False)

        pred_df = pd.DataFrame(preds_dict)
        pred_df.to_csv(out_path / "predictions.csv" , index=False)
    print("Time:", timedelta(seconds=time()-start))
