## About
Code for "Probing for Reading Times".

## Environment
```
# Python 3.11.6
pip install -r requirements.txt
```

## Data
### Provo
Download Provo from the [Open Science Framework](https://osf.io/sjefs)
Preprocess with `preprocess_provo.py`
Score with `score_embed.py`
Compute information value with `score_infovalue.py` and logit lens with `score_logitlens.py`
Convert to word level with `tok2w_provo.py`
Tune with `tune_reg.py`
Cross-validate with `provo_crossval.py`

### MECO
Download MECO from [https://github.com/rycolab/context-reading-time/tree/main/merged_data_no_zero](https://github.com/rycolab/context-reading-time/tree/main/merged_data_no_zero)
Use `meco_trials in supplementary material for scoring`
Score with `score_embed.py`
Compute information value with `score_infovalue.py` and logit lens with `score_logitlens.py`
Convert to word level with `tok2w_meco.py`
Tune with `tune_meco.py`
Cross-validate with `meco_crossval.py`

## Important Note
Work in progress! Stay tuned for feature combinations and a generally tidier version of the code.

## Citation
```
@inproceedings{tsipidi-etal-2026-probing,
    title = "Probing for Reading Times",
    author = "Tsipidi, Eleftheria  and
      Kiegeland, Samuel  and
      Re, Francesco Ignazio  and
      Xu, Tianyang  and
      Giulianelli, Mario  and
      Stanczak, Karolina  and
      Cotterell, Ryan",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.575/",
    doi = "10.18653/v1/2026.acl-long.575",
    pages = "12618--12642",
    ISBN = "979-8-89176-390-6",
    abstract = "Probing has shown that language model representations encode rich linguistic information, but it remains unclear whether they also capture cognitive signals about human processing. In this work, we probe language model representations for human reading times. Using regularized linear regression on two eye-tracking corpora spanning five languages (English, Greek, Hebrew, Russian, and Turkish), we compare the representations from every model layer against scalar predictors{---}surprisal, information value, and logit-lens surprisal. We find that the representations from early layers outperform surprisal in predicting early-pass measures such as first fixation and gaze duration. The concentration of predictive power in the early layers suggests that human-like processing signatures are captured by low-level structural or lexical representations, pointing to a functional alignment between model depth and the temporal stages of human reading. In contrast, for late-pass measures such as total reading time, scalar surprisal remains superior, despite its being a much more compressed representation. We also observe performance gains when using both surprisal and early-layer representations. Overall, we find that the best-performing predictor varies strongly depending on the language and eye-tracking measure."
}
```
