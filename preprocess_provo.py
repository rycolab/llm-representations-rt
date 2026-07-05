import argparse
import random
import string
import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path


WHITELIST = string.ascii_letters + string.digits + ' ' + '!"#$%&\'()*+,-.:;?/'

provo_measures = {
    "IA_FIRST_FIXATION_DURATION": "First Fixation Duration",
    "IA_FIRST_RUN_DWELL_TIME": "Gaze duration",
    "IA_REGRESSION_PATH_DURATION": "Go-Past Time",
    "IA_DWELL_TIME": "Total Reading Time"    
}


def get_problem_words(words, whitelist):
    problem_words = []
    for word in words:
        for ch in word:
            if ch not in whitelist:
                problem_words.append(word)
                break
    return problem_words


def fix_provo(df):
    columns = df.columns.tolist()
    subjects = df["Participant_ID"].unique().tolist()
    ndf = []
    for subject in subjects:
        sdf = df[df["Participant_ID"]==subject]
        sdf = fix_provo_subject(sdf, columns)
        ndf.extend(sdf)
    return pd.concat(ndf).reset_index(drop=True)


def fix_provo_subject(df, columns):
    # returns list of dataframes. Problem texts are 18 and 55
    before_18 = df[df["Text_ID"]<18]
    df_19_54 = df[(df["Text_ID"]>18) & (df["Text_ID"]<55)]
    
    df18 = df[df["Text_ID"]==18].copy().reset_index(drop=True)
    df18.at[2, 'Word_Number'] = '51'

    df55 = df[df["Text_ID"]==55].copy().reset_index(drop=True)
    df55.at[7, "Sentence_Number"] = '1'
    df55.at[7, "Word_In_Sentence_Number"] = '9'
    df55.at[7, "Word_Length"] = '6'
    df55__10 = get_missing55(df55, columns)
    return [before_18, df18[:2], df18[3:50], df18[2:3], df18[50:], df_19_54, df55[:8], df55__10, df55[8:]]


def get_missing55(df55, columns):
    vals = [
        df55.at[0, 'RECORDING_SESSION_LABEL'],
        df55.at[0, 'Participant_ID'],
        'QID2694',
        55,
        '10',
        '1',
        '10',
        'a',
        'a',
        '1'
    ]
    num_missing = len(columns) - len(vals)
    return pd.DataFrame([vals + ['NA' for i in range(num_missing)]], columns=columns)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir", type=str, help="Directory for input and output files")
    args = parser.parse_args()

    fpath = Path(args.fdir)


    ## Clean Provo Predictability Norms
    pn = pd.read_csv(fpath / "Provo_Corpus-Predictability_Norms.csv", keep_default_na=False, encoding = "ISO-8859-1")

    ## how we got the problem words
    # all_docs = pn["Text"].unique().tolist()
    # all_text = ' '.join(all_docs)
    # all_words = set(all_text.split())
    # problem_words = get_problem_words(all_words, WHITELIST)

    problem_words = ['bondsÕ', 'Ñ ', 'womenÕs', 'doesnÕt']
    corrections = ["bonds'", "", "women's", "doesn't"]
    fix_map = dict(zip(problem_words, corrections))

    raw_text = pn["Text"].tolist()
    cleaned_text = []
    for doc in raw_text:
        for word in problem_words:
            doc = doc.replace(word, fix_map[word])
        cleaned_text.append(doc)

    pn["Text"] = cleaned_text
    pn.to_csv(fpath / "provo_pn_clean.csv", index=False)

    out_dir = fpath / "provo_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = pn["Text_ID"].unique().tolist()
    docs = pn["Text"].unique().tolist()

    for doc_id, doc in zip(doc_ids, docs):
        with open(out_dir / f"{doc_id:02}.txt", "w") as fw:
            fw.write(doc)


    ## Clean Provo Eye Tracking

    et = pd.read_csv(fpath / "Provo_Corpus-Eyetracking_Data.csv", keep_default_na=False)
    et = fix_provo(et)

    problem_words = ['TRUE', '0.9', 'women?s', 'bonds?']
    corrections = ['true', '90%', "women's", "bonds'"]
    fix_map = dict(zip(problem_words, corrections))

    wlist = et["Word"].tolist()
    new_word = []
    for word in wlist:
        if word in problem_words:
            word = fix_map[word]
        new_word.append(word)
    et["Word"] = new_word

    wlist = et["Word_Cleaned"].tolist()
    new_word = []
    for word in wlist:
        if word in problem_words:
            word = fix_map[word]
        new_word.append(word)
    et["Word_Cleaned"] = new_word
    et.to_csv(fpath / "provo_et_clean.csv", index=False)

