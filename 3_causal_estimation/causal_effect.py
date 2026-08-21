"""
Path-specific effect estimation for LLM/VLM resume scores.

Adapted from PSE-Pulse-Oximetry (https://github.com/reAIM-Lab/PSE-Pulse-Oximetry).

Causal graph variables (with --with_state_region --with_new_name_group, the paper's setting):
    X = sex or race                         treatment / protected attribute
    Z = [race, age] or [sex, age]           baseline confounders
    V = [pred_exp, edu_level_group]         business-necessity mediators
    W = [state_region, first_name_sex,      redlining proxies
         first_name_age_group, surname_race_label]
    Y = score                               outcome

Effect naming
-------------
This file keeps the upstream VDE / WDE names; the paper renames them:

    VDE  ->  BIE   business-necessity indirect effect (the path through V)
    WDE  ->  RIE   redlining indirect effect          (the path through W)

The estimator produces each of them twice (`...1`, `...2`) — one per decomposition
order — plus self-normalized variants (`_sn`). The paper reports the **self-normalized,
order-symmetric** quantity, i.e. the average of the two orders:

    BIE = (Estimated_VDE_sn1 + Estimated_VDE_sn2) / 2
    RIE = (Estimated_WDE_sn1 + Estimated_WDE_sn2) / 2

Grouped name attributes come from `1_data_generation/01-3_name_sampler.ipynb` and are
attached to the attribute table by `1_data_generation/01-4_var_grouping.ipynb`.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import KFold, GridSearchCV
from pathlib import Path

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
HPARAMS = {
    'n_estimators': [5, 10, 20, 50, 100],
    'max_depth': [1, 2, 3, 4, 5],
    'reg_lambda': [0.5, 1, 2, 5],
}

# ---------------------------------------------------------------------------
# Helper functions (adapted from PSE-Pulse-Oximetry/experiments/estimation.py)
# ---------------------------------------------------------------------------

def fit(X, Y, binary=True, **kwargs):
    model_class = xgb.XGBClassifier if binary else xgb.XGBRegressor
    m = model_class(
        tree_method='hist',
        device='cuda:0',
        verbosity=0,
        enable_categorical=True,
        **kwargs,
    ).fit(X, Y)
    return m


def pred(X, m, binary=True, clip=0.0):
    if binary:
        y = m.predict_proba(X)[:, 1]
    else:
        y = m.predict(X)
    if clip:
        y = np.clip(y, a_min=clip, a_max=1 - clip)
    return y


def grid_search(X, Y, binary, hparams):
    model_class = xgb.XGBClassifier if binary else xgb.XGBRegressor
    scoring = 'neg_brier_score' if binary else 'neg_mean_squared_error'
    m = model_class(tree_method='hist', device='cuda:0', verbosity=0, enable_categorical=True)
    gs = GridSearchCV(
        estimator=m, param_grid=hparams,
        scoring=scoring, cv=5, verbose=0,
    )
    gs.fit(X, Y)
    return gs.best_params_


def rel_err(y_pred, y_true):
    if np.abs(y_true) < 1e-8:
        return None
    return 100 * (y_pred - y_true) / np.abs(y_true)


# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------

def load_vars(X_attr='sex', with_state_region=False, with_new_name_group=False, with_state_geo=False, ):
    if with_state_region:
        W_cols = ['state_region']
    elif with_state_geo:
        W_cols = ['state_geo']
    else:      
        W_cols = ['state_male', 'state_female',
       'state_wh', 'state_bl', 'state_as', 'state_age1', 'state_age2',
       'state_age3']
    if with_new_name_group:
        W_cols += ['first_name_age_group', 'first_name_sex', 'surname_race_label']
    else:
        W_cols += ['first_name_age1', 'first_name_age2', 'first_name_age3',
       'first_name_female', 'first_name_male', 'surname_wh', 'surname_bl',
       'surname_as']
    # V_cols = ['pred_exp', 'edu_level']
    V_cols = ['pred_exp', 'edu_level_group']
    if X_attr == 'sex':
        Z_cols = ['race', 'age']
    else:
        Z_cols = ['sex', 'age']
    vars_ = {
        'X': X_attr,
        'W': W_cols,
        'V': V_cols,
        'Z': Z_cols,
    }
    return vars_


# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_data(mode="txt_llm_scores", score_path=None, X0='Female', X_attr='sex', ):
    if score_path is None:
        raise ValueError(
            "score_path is required. Provide a path to a score CSV file, e.g.:\n"
            "  /path/to/scores/"
            "processed_job_data_..._txt_llm_scores_hf_llama-3.1-8b-instruct.csv"
        )
    df_meta = pd.read_csv(
        '/path/to/CausalFair/Resume/job-distribution/processed_job_data_0102_with_exp_pred_with_names_and_grouping.csv'
    )
    # The published dataset uses shorter public column names. Map them back to the names
    # used below; a no-op when the table already carries the pipeline's own names.
    df_meta = df_meta.rename(columns={"region": "state_region",
                                      "exp_year": "pred_exp",
                                      "edu_group": "edu_level_group",
                                      "state": "state_name"})

    df_score = pd.read_csv(score_path)

    if mode == "txt_scores":
        df_score[['score']] = 100 * df_score[['score']].astype(float)
    else:
        df_score[['score']] = df_score[['score']].astype(float)
    if 'vlm' in score_path:
        df_score["id"] = df_score["id"].apply(lambda x: x.replace("_img", ""))
    df_score["with_skills"] = df_score["id"].apply(
        lambda x: 0 if x.endswith("no_skills") else 1
    )
    df_score["id"] = df_score["id"].apply(lambda x: x.replace("_no_skills", ""))
    if 'vlm' not in score_path:
        df_score["resume_format"] = df_score["id"].apply(
            lambda x: "_".join(str(x).split("_")[:-1])
        )
        df_score["resume_format"] = df_score["resume_format"].replace("", "no_demographics")
    else:
        df_score["resume_format"] = "no_demographics" if "no_images" in score_path else "with_demographics_img"
    df_score["id"] = df_score["id"].apply(lambda x: int(x.split("_")[-1]))
    # print(f"Score data loaded: {len(df_score)} rows, unique resume id: {df_score['id'].nunique()}")

    df_meta["id"] = df_meta["id"].astype(int)
    # print(f"Meta data race distribution: {df_meta['race'].value_counts().to_dict()}")
    df = pd.merge(df_score, df_meta, on="id", how="inner")

    without_AIAN = True
    if without_AIAN:
        n_before = len(df)
        df_check = df[df["race"] == "AIAN"]
        # print(f"AIAN unique values check: {df_check['id'].nunique()} unique IDs, {len(df_check)} total rows")
        df = df[df["race"] != "AIAN"] 
        n_after = len(df)
        print(f"Excluding AIAN individuals: dropped {n_before - n_after} rows, {n_after} rows remaining")

    # Feature engineering
    # df.loc[:, "sex_label"] = df["sex"].map({1: "Male", 0: "Female"})
    print(f"X0: {X0}")
    if X_attr == 'sex':
        if X0 != 'Female':
            df['sex'] = 1 - df['sex']  # flip sex so that 1=X0, 0=X1 for analysis

    if X_attr == 'race':
        # Default: 0=White, 1=Non-White
        df['race'] = df['race'].apply(lambda r: 0 if r == 'White' else 1)
        if X0 != 'White':
            df['race'] = 1 - df['race']

    # ------------------------------------------------------------------
    # Build resume-format dict and select the analysis subset
    # ------------------------------------------------------------------
    resume_format_dict = {}
    for (resume_format, with_skills, job_info_version) in (
        df[["resume_format", "with_skills", "job_info_version"]]
        .drop_duplicates()
        .values
    ):
        resume_format_dict[(resume_format, with_skills, job_info_version)] = df[
            (df["resume_format"] == resume_format)
            & (df["with_skills"] == with_skills)
            & (df["job_info_version"] == job_info_version)
        ]
       
    
    # return df_analysis
    return resume_format_dict


def encode_categoricals(df, cols):
    """Convert string columns to pandas Categorical dtype for XGBoost native support."""
    df = df.copy()
    for c in cols:
        if df[c].dtype == object or df[c].dtype.name == 'category':
            print(f"Encoding column '{c}' as categorical")
            df[c] = df[c].astype('category')
    return df


# ---------------------------------------------------------------------------
# Effect computation from fitted nuisance parameters
# (adapted from PSE-Pulse-Oximetry/experiments/estimation.py)
# ---------------------------------------------------------------------------

def compute_effects_from_df(df, X, Y, Y_binary, W_cols):
    # -----------------------------------------------------------------------
    # do(x) nuisance parameters
    # -----------------------------------------------------------------------
    pi_x0 = (df[X] == 0) / (1 - df['px_z'])
    pi_x1 = (df[X] == 1) / df['px_z']
    pi_x0_sn = pi_x0 / np.mean(pi_x0)
    pi_x1_sn = pi_x1 / np.mean(pi_x1)
    Epi_x0 = np.mean(pi_x0)
    Epi_x1 = np.mean(pi_x1)

    # do(x) standard (AIPW)
    hat_Yx0 = pi_x0 * (df[Y] - df['y_zx0']) + df['y_zx0']
    hat_Yx1 = pi_x1 * (df[Y] - df['y_zx1']) + df['y_zx1']
    Estimated_EYx0 = np.mean(hat_Yx0)
    Estimated_EYx1 = np.mean(hat_Yx1)

    # do(x) self-normalized
    hat_Yx0_sn = pi_x0_sn * (df[Y] - df['y_zx0']) + df['y_zx0']
    hat_Yx1_sn = pi_x1_sn * (df[Y] - df['y_zx1']) + df['y_zx1']
    Estimated_EYx0_sn = np.mean(hat_Yx0_sn)
    Estimated_EYx1_sn = np.mean(hat_Yx1_sn)

    # -----------------------------------------------------------------------
    # NDE / NIE nuisance parameters
    # -----------------------------------------------------------------------
    pi2_ne = (df[X] == 1) * (1 - df['px_wvz']) / (df['px_wvz'] * (1 - df['px_z']))
    pi1_ne = (df[X] == 0) / (1 - df['px_z'])
    pi2_ne_sn = pi2_ne / np.mean(pi2_ne)
    pi1_ne_sn = pi1_ne / np.mean(pi1_ne)
    Epi2_ne = np.mean(pi2_ne)
    Epi1_ne = np.mean(pi1_ne)

    # NDE / NIE standard
    Estimated_EYx1WVx0_pi2Y = np.mean(pi2_ne * df[Y])
    Estimated_EYx1WVx0_pi2mu2 = np.mean(pi2_ne * df['mu2_ne'])
    Estimated_EYx1WVx0_pi1mu2 = np.mean(pi1_ne * df['mu2_ne'])
    Estimated_EYx1WVx0_pi1mu1 = np.mean(pi1_ne * df['mu1_ne'])
    Estimated_EYx1WVx0_mu1 = np.mean(df['mu1_ne'])
    Estimated_EYx1WVx0 = (
        (Estimated_EYx1WVx0_pi2Y - Estimated_EYx1WVx0_pi2mu2)
        + (Estimated_EYx1WVx0_pi1mu2 - Estimated_EYx1WVx0_pi1mu1)
        + Estimated_EYx1WVx0_mu1
    )
    Estimated_NDE = Estimated_EYx1WVx0 - Estimated_EYx0
    Estimated_NIE = Estimated_EYx1 - Estimated_EYx1WVx0

    # NDE / NIE self-normalized
    Estimated_EYx1WVx0_pi2Y_sn = np.mean(pi2_ne_sn * df[Y])
    Estimated_EYx1WVx0_pi2mu2_sn = np.mean(pi2_ne_sn * df['mu2_ne'])
    Estimated_EYx1WVx0_pi1mu2_sn = np.mean(pi1_ne_sn * df['mu2_ne'])
    Estimated_EYx1WVx0_pi1mu1_sn = np.mean(pi1_ne_sn * df['mu1_ne'])
    Estimated_EYx1WVx0_sn = (
        (Estimated_EYx1WVx0_pi2Y_sn - Estimated_EYx1WVx0_pi2mu2_sn)
        + (Estimated_EYx1WVx0_pi1mu2_sn - Estimated_EYx1WVx0_pi1mu1_sn)
        + Estimated_EYx1WVx0_mu1
    )
    Estimated_NDE_sn = Estimated_EYx1WVx0_sn - Estimated_EYx0_sn
    Estimated_NIE_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_sn

    # -----------------------------------------------------------------------
    # VDE nuisance parameters
    # -----------------------------------------------------------------------
    pi3_vde = (
        (df[X] == 1)
        * (1 - df['px_wvz']) * df['px_wz']
        / (df['px_wvz'] * (1 - df['px_wz']) * df['px_z'])
    )
    pi2_vde = (
        (df[X] == 0) * df['px_wz']
        / ((1 - df['px_wz']) * df['px_z'])
    )
    pi1_vde = (df[X] == 1) / df['px_z']
    pi3_vde_sn = pi3_vde / np.mean(pi3_vde)
    pi2_vde_sn = pi2_vde / np.mean(pi2_vde)
    pi1_vde_sn = pi1_vde / np.mean(pi1_vde)
    Epi3_vde = np.mean(pi3_vde)
    Epi2_vde = np.mean(pi2_vde)
    Epi1_vde = np.mean(pi1_vde)

    # VDE standard
    Estimated_EYx1Vx0Wx1_pi3Y = np.mean(pi3_vde * df[Y])
    Estimated_EYx1Vx0Wx1_pi3mu3 = np.mean(pi3_vde * df['mu3_vde'])
    Estimated_EYx1Vx0Wx1_pi2mu3 = np.mean(pi2_vde * df['mu3_vde'])
    Estimated_EYx1Vx0Wx1_pi2mu2 = np.mean(pi2_vde * df['mu2_vde'])
    Estimated_EYx1Vx0Wx1_pi1mu2 = np.mean(pi1_vde * df['mu2_vde'])
    Estimated_EYx1Vx0Wx1_pi1mu1 = np.mean(pi1_vde * df['mu1_vde'])
    Estimated_EYx1Vx0Wx1_mu1 = np.mean(df['mu1_vde'])
    Estimated_EYx1Vx0Wx1 = (
        (Estimated_EYx1Vx0Wx1_pi3Y - Estimated_EYx1Vx0Wx1_pi3mu3)
        + (Estimated_EYx1Vx0Wx1_pi2mu3 - Estimated_EYx1Vx0Wx1_pi2mu2)
        + (Estimated_EYx1Vx0Wx1_pi1mu2 - Estimated_EYx1Vx0Wx1_pi1mu1)
        + Estimated_EYx1Vx0Wx1_mu1
    )
    Estimated_VDE1 = Estimated_EYx1 - Estimated_EYx1Vx0Wx1
    Estimated_WDE1 = Estimated_EYx1Vx0Wx1 - Estimated_EYx1WVx0

    # VDE self-normalized
    Estimated_EYx1Vx0Wx1_pi3Y_sn = np.mean(pi3_vde_sn * df[Y])
    Estimated_EYx1Vx0Wx1_pi3mu3_sn = np.mean(pi3_vde_sn * df['mu3_vde'])
    Estimated_EYx1Vx0Wx1_pi2mu3_sn = np.mean(pi2_vde_sn * df['mu3_vde'])
    Estimated_EYx1Vx0Wx1_pi2mu2_sn = np.mean(pi2_vde_sn * df['mu2_vde'])
    Estimated_EYx1Vx0Wx1_pi1mu2_sn = np.mean(pi1_vde_sn * df['mu2_vde'])
    Estimated_EYx1Vx0Wx1_pi1mu1_sn = np.mean(pi1_vde_sn * df['mu1_vde'])
    Estimated_EYx1Vx0Wx1_sn = (
        (Estimated_EYx1Vx0Wx1_pi3Y_sn - Estimated_EYx1Vx0Wx1_pi3mu3_sn)
        + (Estimated_EYx1Vx0Wx1_pi2mu3_sn - Estimated_EYx1Vx0Wx1_pi2mu2_sn)
        + (Estimated_EYx1Vx0Wx1_pi1mu2_sn - Estimated_EYx1Vx0Wx1_pi1mu1_sn)
        + Estimated_EYx1Vx0Wx1_mu1
    )
    Estimated_VDE_sn1 = Estimated_EYx1_sn - Estimated_EYx1Vx0Wx1_sn
    Estimated_WDE_sn1 = Estimated_EYx1Vx0Wx1_sn - Estimated_EYx1WVx0_sn

    # -----------------------------------------------------------------------
    # WDE  (W-specific direct effect, not in original PSE repo)
    # -----------------------------------------------------------------------
    # fix, p(W|X=0,Z) / p(W|X,Z) term should be multiplied,
    pi3_wde = (
        ((df[X] == 1) * (1 - df['px_wz'])) / ((1-df['px_z']) * df['px_wz'])
    )

    # π₀²: P(W|x0,Z)/P(W|X,Z) · 1[X=x1] / P(X|Z)
    # When X=1, Bayes trick gives: (1 - px_wz) * px_z / ((1 - px_z) * px_wz)
    # Then times 1/(px_z), the px_z cancels:
    pi2_wde = (
        (df[X] == 1) * (1 - df['px_wz'])
        / ((1 - df['px_z']) * df['px_wz'])
    )

    # π₀¹: 1[X=x0] / P(X|Z) = 1[X=0] / (1 - px_z)
    pi1_wde = (df[X] == 0) / (1 - df['px_z'])   
    pi3_wde_sn = pi3_wde / np.mean(pi3_wde)
    pi2_wde_sn = pi2_wde / np.mean(pi2_wde)
    pi1_wde_sn = pi1_wde / np.mean(pi1_wde)
    Epi3_wde = np.mean(pi3_wde)
    Epi2_wde = np.mean(pi2_wde)
    Epi1_wde = np.mean(pi1_wde)     


    # WDE standard
    Estimated_EYx1Vx1Wx0_pi3Y = np.mean(pi3_wde * df[Y])
    Estimated_EYx1Vx1Wx0_pi3mu3 = np.mean(pi3_wde * df['mu3_wde'])
    Estimated_EYx1Vx1Wx0_pi2mu3 = np.mean(pi2_wde * df['mu3_wde'])
    Estimated_EYx1Vx1Wx0_pi2mu2 = np.mean(pi2_wde * df['mu2_wde'])
    Estimated_EYx1Vx1Wx0_pi1mu2 = np.mean(pi1_wde * df['mu2_wde'])
    Estimated_EYx1Vx1Wx0_pi1mu1 = np.mean(pi1_wde * df['mu1_wde'])
    Estimated_EYx1Vx1Wx0_mu1 = np.mean(df['mu1_wde'])
    Estimated_EYx1Vx1Wx0 = (
        (Estimated_EYx1Vx1Wx0_pi3Y - Estimated_EYx1Vx1Wx0_pi3mu3)
        + (Estimated_EYx1Vx1Wx0_pi2mu3 - Estimated_EYx1Vx1Wx0_pi2mu2)
        + (Estimated_EYx1Vx1Wx0_pi1mu2 - Estimated_EYx1Vx1Wx0_pi1mu1)
        + Estimated_EYx1Vx1Wx0_mu1
    )
    Estimated_WDE2 = Estimated_EYx1 - Estimated_EYx1Vx1Wx0
    Estimated_VDE2 = Estimated_EYx1Vx1Wx0 - Estimated_EYx1WVx0
    
    # WDE self-normalized
    Estimated_EYx1Vx1Wx0_pi3Y_sn = np.mean(pi3_wde_sn * df[Y])
    Estimated_EYx1Vx1Wx0_pi3mu3_sn = np.mean(pi3_wde_sn * df['mu3_wde'])
    Estimated_EYx1Vx1Wx0_pi2mu3_sn = np.mean(pi2_wde_sn * df['mu3_wde'])
    Estimated_EYx1Vx1Wx0_pi2mu2_sn = np.mean(pi2_wde_sn * df['mu2_wde'])
    Estimated_EYx1Vx1Wx0_pi1mu2_sn = np.mean(pi1_wde_sn * df['mu2_wde'])
    Estimated_EYx1Vx1Wx0_pi1mu1_sn = np.mean(pi1_wde_sn * df['mu1_wde'])
    Estimated_EYx1Vx1Wx0_sn = (
        (Estimated_EYx1Vx1Wx0_pi3Y_sn - Estimated_EYx1Vx1Wx0_pi3mu3_sn)
        + (Estimated_EYx1Vx1Wx0_pi2mu3_sn - Estimated_EYx1Vx1Wx0_pi2mu2_sn)
        + (Estimated_EYx1Vx1Wx0_pi1mu2_sn - Estimated_EYx1Vx1Wx0_pi1mu1_sn)
        + Estimated_EYx1Vx1Wx0_mu1
    )
    Estimated_WDE_sn2 = Estimated_EYx1_sn - Estimated_EYx1Vx1Wx0_sn
    Estimated_VDE_sn2 = Estimated_EYx1Vx1Wx0_sn - Estimated_EYx1WVx0_sn
    # -----------------------------------------------------------------------
    # NIE* (using V,Z only -- no W)
    # -----------------------------------------------------------------------
    pi2_star = (df[X] == 1) * (1 - df['px_vz']) / (df['px_vz'] * (1 - df['px_z']))
    pi1_star = (df[X] == 0) / (1 - df['px_z'])
    pi2_star_sn = pi2_star / np.mean(pi2_star)
    pi1_star_sn = pi1_star / np.mean(pi1_star)

    Estimated_EYx1WVx0_pi2Y_star = np.mean(pi2_star * df[Y])
    Estimated_EYx1WVx0_pi2mu2_star = np.mean(pi2_star * df['mu2_ne*'])
    Estimated_EYx1WVx0_pi1mu2_star = np.mean(pi1_star * df['mu2_ne*'])
    Estimated_EYx1WVx0_pi1mu1_star = np.mean(pi1_star * df['mu1_ne*'])
    Estimated_EYx1WVx0_mu1_star = np.mean(df['mu1_ne*'])
    Estimated_EYx1WVx0_star = (
        (Estimated_EYx1WVx0_pi2Y_star - Estimated_EYx1WVx0_pi2mu2_star)
        + (Estimated_EYx1WVx0_pi1mu2_star - Estimated_EYx1WVx0_pi1mu1_star)
        + Estimated_EYx1WVx0_mu1_star
    )
    Estimated_NDE_star = Estimated_EYx1WVx0_star - Estimated_EYx0
    Estimated_NIE_star = Estimated_EYx1 - Estimated_EYx1WVx0_star

    Estimated_EYx1WVx0_pi2Y_star_sn = np.mean(pi2_star_sn * df[Y])
    Estimated_EYx1WVx0_pi2mu2_star_sn = np.mean(pi2_star_sn * df['mu2_ne*'])
    Estimated_EYx1WVx0_pi1mu2_star_sn = np.mean(pi1_star_sn * df['mu2_ne*'])
    Estimated_EYx1WVx0_pi1mu1_star_sn = np.mean(pi1_star_sn * df['mu1_ne*'])
    Estimated_EYx1WVx0_star_sn = (
        (Estimated_EYx1WVx0_pi2Y_star_sn - Estimated_EYx1WVx0_pi2mu2_star_sn)
        + (Estimated_EYx1WVx0_pi1mu2_star_sn - Estimated_EYx1WVx0_pi1mu1_star_sn)
        + Estimated_EYx1WVx0_mu1_star
    )
    Estimated_NDE_star_sn = Estimated_EYx1WVx0_star_sn - Estimated_EYx0_sn
    Estimated_NIE_star_sn = Estimated_EYx1_sn - Estimated_EYx1WVx0_star_sn

    # -----------------------------------------------------------------------
    # Biased TE (conditioning on all post-treatment variables)
    # -----------------------------------------------------------------------
    pi_x0_biased = (df[X] == 0) / (1 - df['px_wvz'])
    pi_x1_biased = (df[X] == 1) / df['px_wvz']
    hat_Yx0_biased = pi_x0_biased * (df[Y] - df['y_wvzx0']) + df['y_wvzx0']
    hat_Yx1_biased = pi_x1_biased * (df[Y] - df['y_wvzx1']) + df['y_wvzx1']
    Estimated_EYx0_biased = np.mean(hat_Yx0_biased)
    Estimated_EYx1_biased = np.mean(hat_Yx1_biased)

    # Total effects
    Estimated_TE = Estimated_EYx1 - Estimated_EYx0
    Estimated_TE_sn = Estimated_EYx1_sn - Estimated_EYx0_sn
    Estimated_TE_biased = Estimated_EYx1_biased - Estimated_EYx0_biased

    # -----------------------------------------------------------------------
    # Assemble results
    # -----------------------------------------------------------------------
    results = {
        'Estimated_EYx0': [Estimated_EYx0],
        'Estimated_EYx1': [Estimated_EYx1],
        'Estimated_EYx0_sn': [Estimated_EYx0_sn],
        'Estimated_EYx1_sn': [Estimated_EYx1_sn],
        'Estimated_EYx1WVx0': [Estimated_EYx1WVx0],
        'Estimated_EYx1Vx0Wx1': [Estimated_EYx1Vx0Wx1],
        'Estimated_EYx1Vx1Wx0': [Estimated_EYx1Vx1Wx0],
        'Estimated_EYx1WVx0*': [Estimated_EYx1WVx0_star],
        'Estimated_EYx1WVx0_sn': [Estimated_EYx1WVx0_sn],
        'Estimated_EYx1Vx0Wx1_sn': [Estimated_EYx1Vx0Wx1_sn],
        'Estimated_EYx1Vx1Wx0_sn': [Estimated_EYx1Vx1Wx0_sn],
        'Estimated_EYx1WVx0*_sn': [Estimated_EYx1WVx0_star_sn],
        # Total effect
        'Estimated_TE': [Estimated_TE],
        'Estimated_TE_sn': [Estimated_TE_sn],
        'Estimated_TE_biased': [Estimated_TE_biased],
        # Path-specific effects
        'Estimated_NDE': [Estimated_NDE],
        'Estimated_NIE': [Estimated_NIE],
        'Estimated_VDE1': [Estimated_VDE1],
        'Estimated_WDE1': [Estimated_WDE1],
        'Estimated_VDE2': [Estimated_VDE2],
        'Estimated_WDE2': [Estimated_WDE2],
        'Estimated_NDE_sn': [Estimated_NDE_sn],
        'Estimated_NIE_sn': [Estimated_NIE_sn],
        'Estimated_VDE_sn1': [Estimated_VDE_sn1],
        'Estimated_VDE_sn2': [Estimated_VDE_sn2],
        'Estimated_WDE_sn1': [Estimated_WDE_sn1],
        'Estimated_WDE_sn2': [Estimated_WDE_sn2],
        'Estimated_NDE*': [Estimated_NDE_star],
        'Estimated_NIE*': [Estimated_NIE_star],
        'Estimated_NDE*_sn': [Estimated_NDE_star_sn],
        'Estimated_NIE*_sn': [Estimated_NIE_star_sn],
        # Nuisance parameter diagnostics
        'Epi_x0': [Epi_x0],
        'Epi_x1': [Epi_x1],
        'Epi2_ne': [Epi2_ne],
        'Epi1_ne': [Epi1_ne],
        'Epi3_vde': [Epi3_vde],
        'Epi2_vde': [Epi2_vde],
        'Epi1_vde': [Epi1_vde],
        'Epi3_wde': [Epi3_wde],
        'Epi2_wde': [Epi2_wde],
        'Epi1_wde': [Epi1_wde],
    }
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main estimation routine
# ---------------------------------------------------------------------------

def estimate_effects(
    sample, all_df, X, Y, W_cols, V_cols, Z_cols, Y_binary,
    N, clip=0.0, K=5, bootstraps=30, n_mc=1000,
    sample_kwargs=None, hparam_path=None,
):
    assert (sample is not None) or (all_df is not None)
    gt_effects = None
    if sample is None:
        N = [len(all_df)]

    model_list = [
        'px_z', 'px_wz', 'px_vz', 'px_wvz',
        'y_zx0', 'y_zx1',
        'y_wvzx0', 'y_wvzx1',
        'y_wzx0', 'y_wzx1',
        'mu2_ne', 'mu1_ne',
        'mu3_vde', 'mu2_vde', 'mu1_vde',
        'mu3_wde', 'mu2_wde', 'mu1_wde',
    ]
    if all_df is not None:
        model_list += ['y_vzx0', 'y_vzx1', 'mu2_ne*', 'mu1_ne*']

    if hparam_path and os.path.exists(hparam_path):
        print('Loading hparams from', hparam_path)
        with open(hparam_path, 'r') as f:
            hp_dict = json.load(f)
    else:
        print('Fitting hparams and saving to', hparam_path)
        hp_dict = {}

    search_df = all_df.copy()
    aux_cols = {c: None for c in ['mu2_ne', 'mu2_ne*', 'mu3_vde', 'mu2_vde']}
    aux_df = pd.DataFrame(aux_cols, index=search_df.index)
    search_df = pd.concat([search_df, aux_df], axis=1)
    search_df = search_df.reset_index(drop=True)
    search_df_x0 = search_df[search_df[X] == 0]
    search_df_x1 = search_df[search_df[X] == 1]
    idxs = np.arange(len(search_df))

    # -------------------------------------------------------------------
    # Hyperparameter tuning (propensity models)
    # -------------------------------------------------------------------
    print('Fitting propensity hparams')
    for m in tqdm(['px_z', 'px_wz', 'px_vz', 'px_wvz']):
        if m in hp_dict:
            continue
        cols = list(Z_cols)
        cond = m.split('_')[1]
        if 'w' in cond:
            cols += W_cols
        if 'v' in cond:
            cols += V_cols
        hp = grid_search(search_df[cols], search_df[X], True, HPARAMS)
        hp_dict[m] = hp

    # -------------------------------------------------------------------
    # Hyperparameter tuning (outcome models)
    # -------------------------------------------------------------------
    print('Fitting outcome hparams')
    if 'y_zx0' not in hp_dict:
        hp_dict['y_zx0'] = grid_search(
            search_df_x0[Z_cols], search_df_x0[Y], Y_binary, HPARAMS)
    if 'y_zx1' not in hp_dict:
        hp_dict['y_zx1'] = grid_search(
            search_df_x1[Z_cols], search_df_x1[Y], Y_binary, HPARAMS)
    if 'y_wvzx0' not in hp_dict:
        hp_dict['y_wvzx0'] = grid_search(
            search_df_x0[Z_cols + W_cols + V_cols], search_df_x0[Y], Y_binary, HPARAMS)
    if 'y_wvzx1' not in hp_dict:
        hp_dict['y_wvzx1'] = grid_search(
            search_df_x1[Z_cols + W_cols + V_cols], search_df_x1[Y], Y_binary, HPARAMS)
    if 'y_wzx0' not in hp_dict:
        hp_dict['y_wzx0'] = grid_search(
            search_df_x0[Z_cols + W_cols], search_df_x0[Y], Y_binary, HPARAMS)
    if 'y_wzx1' not in hp_dict:
        hp_dict['y_wzx1'] = grid_search(
            search_df_x1[Z_cols + W_cols], search_df_x1[Y], Y_binary, HPARAMS)
    if all_df is not None:
        if 'y_vzx0' not in hp_dict:
            hp_dict['y_vzx0'] = grid_search(
                search_df_x0[Z_cols + V_cols], search_df_x0[Y], Y_binary, HPARAMS)
        if 'y_vzx1' not in hp_dict:
            hp_dict['y_vzx1'] = grid_search(
                search_df_x1[Z_cols + V_cols], search_df_x1[Y], Y_binary, HPARAMS)

    # -------------------------------------------------------------------
    # Hyperparameter tuning (nested outcome models)
    # -------------------------------------------------------------------
    print('Fitting nested outcome hparams')

    # -- NDE nested --
    if 'mu2_zx0' not in hp_dict:
        np.random.seed(1001)
        np.random.shuffle(idxs)
        mu, ns = np.array_split(idxs, 2)
        df_mu, df_ns = search_df.iloc[mu], search_df.iloc[ns]
        (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
        (_, df_ns_x0), (_, df_ns_x1) = df_ns.groupby(df_ns[X])
        y_wvzx1 = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y],
                       binary=Y_binary, **hp_dict['y_wvzx1'])
        df_ns_x0 = df_ns_x0.copy()
        df_ns_x0['mu2_ne'] = pred(df_ns_x0[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)
        hp = grid_search(df_ns_x0[Z_cols], df_ns_x0['mu2_ne'], False, HPARAMS)
        hp_dict['mu2_zx0'] = hp

    # -- NDE* nested (V,Z only) --
    if all_df is not None and 'mu2_zx0*' not in hp_dict:
        np.random.seed(1001)
        idxs_copy = np.arange(len(search_df))
        np.random.shuffle(idxs_copy)
        mu, ns = np.array_split(idxs_copy, 2)
        df_mu, df_ns = search_df.iloc[mu], search_df.iloc[ns]
        (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
        (_, df_ns_x0), (_, df_ns_x1) = df_ns.groupby(df_ns[X])
        y_vzx1 = fit(df_mu_x1[V_cols + Z_cols], df_mu_x1[Y],
                      binary=Y_binary, **hp_dict['y_vzx1'])
        df_ns_x0 = df_ns_x0.copy()
        df_ns_x0['mu2_ne*'] = pred(df_ns_x0[V_cols + Z_cols], y_vzx1, binary=Y_binary)
        hp = grid_search(df_ns_x0[Z_cols], df_ns_x0['mu2_ne*'], False, HPARAMS)
        hp_dict['mu2_zx0*'] = hp

    # -- VDE nested (triple split) --
    if 'mu3_wzx0' not in hp_dict or 'mu2_zx1' not in hp_dict:
        np.random.seed(1043)
        idxs_copy = np.arange(len(search_df))
        np.random.shuffle(idxs_copy)
        mu, ns1, ns2 = np.array_split(idxs_copy, 3)
        df_mu = search_df.iloc[mu]
        df_ns1 = search_df.iloc[ns1]
        df_ns2 = search_df.iloc[ns2]
        (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
        (_, df_ns1_x0), (_, df_ns1_x1) = df_ns1.groupby(df_ns1[X])
        (_, df_ns2_x0), (_, df_ns2_x1) = df_ns2.groupby(df_ns2[X])
        y_wvzx1 = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y],
                       binary=Y_binary, **hp_dict['y_wvzx1'])
        df_ns1_x0 = df_ns1_x0.copy()
        df_ns1_x0['mu3_vde'] = pred(df_ns1_x0[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)
        if 'mu3_wzx0' not in hp_dict:
            hp = grid_search(df_ns1_x0[W_cols + Z_cols], df_ns1_x0['mu3_vde'], False, HPARAMS)
            hp_dict['mu3_wzx0'] = hp

        mu3_wzx0 = fit(df_ns1_x0[W_cols + Z_cols], df_ns1_x0['mu3_vde'],
                        binary=False, **hp_dict['mu3_wzx0'])
        df_ns2_x1 = df_ns2_x1.copy()
        df_ns2_x1['mu2_vde'] = pred(df_ns2_x1[W_cols + Z_cols], mu3_wzx0, binary=False)
        if 'mu2_zx1' not in hp_dict:
            hp = grid_search(df_ns2_x1[Z_cols], df_ns2_x1['mu2_vde'], False, HPARAMS)
            hp_dict['mu2_zx1'] = hp
            
    # -- WDE nested (triple split) --
    if 'mu3_wzx1' not in hp_dict or 'mu2_zx0' not in hp_dict:
        np.random.seed(1043)
        idxs_copy = np.arange(len(search_df))
        np.random.shuffle(idxs_copy)
        mu, ns1, ns2 = np.array_split(idxs_copy, 3)
        df_mu = search_df.iloc[mu]
        df_ns1 = search_df.iloc[ns1]
        df_ns2 = search_df.iloc[ns2]
        (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
        (_, df_ns1_x0), (_, df_ns1_x1) = df_ns1.groupby(df_ns1[X])
        (_, df_ns2_x0), (_, df_ns2_x1) = df_ns2.groupby(df_ns2[X])
        y_wvzx1 = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y],
                       binary=Y_binary, **hp_dict['y_wvzx1'])
        df_ns1_x1 = df_ns1_x1.copy()
        df_ns1_x1['mu3_wde'] = pred(df_ns1_x1[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)
        if 'mu3_wzx1' not in hp_dict:
            hp = grid_search(df_ns1_x1[W_cols + Z_cols], df_ns1_x1['mu3_wde'], False, HPARAMS)
            hp_dict['mu3_wzx1'] = hp

        mu3_wzx1 = fit(df_ns1_x1[W_cols + Z_cols], df_ns1_x1['mu3_wde'],
                        binary=False, **hp_dict['mu3_wzx1'])
        df_ns2_x0 = df_ns2_x0.copy()
        df_ns2_x0['mu2_wde'] = pred(df_ns2_x0[W_cols + Z_cols], mu3_wzx1, binary=False)
        if 'mu2_zx0' not in hp_dict:
            hp = grid_search(df_ns2_x0[Z_cols], df_ns2_x0['mu2_wde'], False, HPARAMS)
            hp_dict['mu2_zx0'] = hp

    # Save hyperparameters
    if hparam_path:
        os.makedirs(os.path.dirname(hparam_path) or '.', exist_ok=True)
        with open(hparam_path, 'w') as f:
            json.dump(hp_dict, f, indent=4)

    # -------------------------------------------------------------------
    # Main estimation loop (cross-fitting + bootstraps)
    # -------------------------------------------------------------------
    results_df = pd.DataFrame()
    aux_cols = {c: None for c in model_list}
    synthetic = (sample is not None)

    for n in N:
        all_idxs = np.arange(n)
        print('Sample size:', n)
        for i in tqdm(range(bootstraps)):
            folds = KFold(n_splits=K, shuffle=True, random_state=i)
            if sample is not None:
                df = sample(n=n, seed=i, **sample_kwargs)
                df = df[[X, Y] + W_cols + V_cols + Z_cols]
            else:
                df = all_df.copy()
                # with replacement sampling for df
                df = df.sample(n=n, replace=True, random_state=i).reset_index(drop=True)

            aux_df = pd.DataFrame(aux_cols, index=df.index)
            df = pd.concat([df, aux_df], axis=1)
            df = df.reset_index(drop=True)

            for tr, ts in folds.split(all_idxs):
                df_tr, df_ts = df.iloc[tr], df.iloc[ts]
                (_, df_tr_x0), (_, df_tr_x1) = df_tr.groupby(df_tr[X])

                # -- Propensity models --
                px_z = fit(df_tr[Z_cols], df_tr[X], **hp_dict['px_z'])
                px_wz = fit(df_tr[W_cols + Z_cols], df_tr[X], **hp_dict['px_wz'])
                px_vz = fit(df_tr[V_cols + Z_cols], df_tr[X], **hp_dict['px_vz'])
                px_wvz = fit(df_tr[W_cols + V_cols + Z_cols], df_tr[X], **hp_dict['px_wvz'])
                df.loc[ts, 'px_z'] = pred(df_ts[Z_cols], px_z, clip=clip)
                df.loc[ts, 'px_wz'] = pred(df_ts[W_cols + Z_cols], px_wz, clip=clip)
                df.loc[ts, 'px_vz'] = pred(df_ts[V_cols + Z_cols], px_vz, clip=clip)
                df.loc[ts, 'px_wvz'] = pred(df_ts[W_cols + V_cols + Z_cols], px_wvz, clip=clip)

                # -- Outcome models --
                y_zx0 = fit(df_tr_x0[Z_cols], df_tr_x0[Y], binary=Y_binary, **hp_dict['y_zx0'])
                y_zx1 = fit(df_tr_x1[Z_cols], df_tr_x1[Y], binary=Y_binary, **hp_dict['y_zx1'])
                y_wvzx0 = fit(df_tr_x0[W_cols + V_cols + Z_cols], df_tr_x0[Y],
                              binary=Y_binary, **hp_dict['y_wvzx0'])
                y_wvzx1 = fit(df_tr_x1[W_cols + V_cols + Z_cols], df_tr_x1[Y],
                              binary=Y_binary, **hp_dict['y_wvzx1'])
                df.loc[ts, 'y_zx0'] = pred(df_ts[Z_cols], y_zx0, binary=Y_binary)
                df.loc[ts, 'y_zx1'] = pred(df_ts[Z_cols], y_zx1, binary=Y_binary)
                df.loc[ts, 'y_wvzx0'] = pred(df_ts[W_cols + V_cols + Z_cols], y_wvzx0, binary=Y_binary)
                df.loc[ts, 'y_wvzx1'] = pred(df_ts[W_cols + V_cols + Z_cols], y_wvzx1, binary=Y_binary)

                # -------------------------------------------------------
                # NDE / NIE nested (2-way split)
                # -------------------------------------------------------
                np.random.seed(i)
                tr_copy = tr.copy()
                np.random.shuffle(tr_copy)
                mu, ns = np.array_split(tr_copy, 2)
                df_mu, df_ns = df.iloc[mu], df.iloc[ns]
                (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
                (_, df_ns_x0), (_, df_ns_x1) = df_ns.groupby(df_ns[X])

                # E[Y | W, V, Z, X=1] regressed on Z at X=0
                y_wvzx1_ = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y],
                               binary=Y_binary, **hp_dict['y_wvzx1'])
                df_ns_x0 = df_ns_x0.copy()
                df_ns_x0['mu2_ne'] = pred(df_ns_x0[W_cols + V_cols + Z_cols], y_wvzx1_, binary=Y_binary)
                df.loc[ts, 'mu2_ne'] = pred(df_ts[W_cols + V_cols + Z_cols], y_wvzx1_, binary=Y_binary)
                mu1_zx0 = fit(df_ns_x0[Z_cols], df_ns_x0['mu2_ne'], binary=False, **hp_dict['mu2_zx0'])
                df.loc[ts, 'mu1_ne'] = pred(df_ts[Z_cols], mu1_zx0, binary=False)

                # NIE* (V, Z only)
                if all_df is not None:
                    y_vzx1_ = fit(df_mu_x1[V_cols + Z_cols], df_mu_x1[Y],
                                  binary=Y_binary, **hp_dict['y_vzx1'])
                    df_ns_x0['mu2_ne*'] = pred(df_ns_x0[V_cols + Z_cols], y_vzx1_, binary=Y_binary)
                    df.loc[ts, 'mu2_ne*'] = pred(df_ts[V_cols + Z_cols], y_vzx1_, binary=Y_binary)
                    mu1_zx0_star = fit(df_ns_x0[Z_cols], df_ns_x0['mu2_ne*'],
                                       binary=False, **hp_dict['mu2_zx0*'])
                    df.loc[ts, 'mu1_ne*'] = pred(df_ts[Z_cols], mu1_zx0_star, binary=False)
                    if Y_binary:
                        df.loc[ts, 'mu1_ne*'] = np.clip(df.loc[ts, 'mu1_ne*'], 0, 1)

                # -------------------------------------------------------
                # VDE nested (3-way split)
                # -------------------------------------------------------
                np.random.seed(i + 42)
                tr_copy2 = tr.copy()
                np.random.shuffle(tr_copy2)
                mu, ns1, ns2 = np.array_split(tr_copy2, 3)
                df_mu = df.iloc[mu]
                df_ns1, df_ns2 = df.iloc[ns1], df.iloc[ns2]
                (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
                (_, df_ns1_x0), (_, df_ns1_x1) = df_ns1.groupby(df_ns1[X])
                (_, df_ns2_x0), (_, df_ns2_x1) = df_ns2.groupby(df_ns2[X])

                y_wvzx1_v = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y],
                                binary=Y_binary, **hp_dict['y_wvzx1'])
                df_ns1_x0 = df_ns1_x0.copy()
                df_ns1_x0['mu3_vde'] = pred(df_ns1_x0[W_cols + V_cols + Z_cols], y_wvzx1_v, binary=Y_binary)
                df.loc[ts, 'mu3_vde'] = pred(df_ts[W_cols + V_cols + Z_cols], y_wvzx1_v, binary=Y_binary)

                mu1_wzx0 = fit(df_ns1_x0[W_cols + Z_cols], df_ns1_x0['mu3_vde'],
                               binary=False, **hp_dict['mu3_wzx0'])
                df_ns2_x1 = df_ns2_x1.copy()
                df_ns2_x1['mu2_vde'] = pred(df_ns2_x1[W_cols + Z_cols], mu1_wzx0, binary=False)
                df.loc[ts, 'mu2_vde'] = pred(df_ts[W_cols + Z_cols], mu1_wzx0, binary=False)
                if Y_binary:
                    df_ns2_x1['mu2_vde'] = np.clip(df_ns2_x1['mu2_vde'], 0, 1)
                    df.loc[ts, 'mu2_vde'] = np.clip(df.loc[ts, 'mu2_vde'], 0, 1)

                mu2_zx1 = fit(df_ns2_x1[Z_cols], df_ns2_x1['mu2_vde'],
                              binary=False, **hp_dict['mu2_zx1'])
                df.loc[ts, 'mu1_vde'] = pred(df_ts[Z_cols], mu2_zx1, binary=False)
                if Y_binary:
                    df.loc[ts, 'mu1_vde'] = np.clip(df.loc[ts, 'mu1_vde'], 0, 1)
                    
            
                # -------------------------------------------------------
                # WDE nested (3-way split)
                # -------------------------------------------------------
                np.random.seed(i + 42)
                tr_copy2 = tr.copy()
                np.random.shuffle(tr_copy2)
                mu, ns1, ns2 = np.array_split(tr_copy2, 3)
                df_mu = df.iloc[mu]
                df_ns1, df_ns2 = df.iloc[ns1], df.iloc[ns2]
                (_, df_mu_x0), (_, df_mu_x1) = df_mu.groupby(df_mu[X])
                (_, df_ns1_x0), (_, df_ns1_x1) = df_ns1.groupby(df_ns1[X])
                (_, df_ns2_x0), (_, df_ns2_x1) = df_ns2.groupby(df_ns2[X])

                y_wvzx1_v = fit(df_mu_x1[W_cols + V_cols + Z_cols], df_mu_x1[Y],
                                binary=Y_binary, **hp_dict['y_wvzx1'])
                df_ns1_x1 = df_ns1_x1.copy()
                df_ns1_x1['mu3_wde'] = pred(df_ns1_x1[W_cols + V_cols + Z_cols], y_wvzx1_v, binary=Y_binary)
                df.loc[ts, 'mu3_wde'] = pred(df_ts[W_cols + V_cols + Z_cols], y_wvzx1_v, binary=Y_binary)

                mu1_wzx1 = fit(df_ns1_x1[W_cols + Z_cols], df_ns1_x1['mu3_wde'],
                               binary=False, **hp_dict['mu3_wzx1'])
                df_ns2_x0 = df_ns2_x0.copy()
                df_ns2_x0['mu2_wde'] = pred(df_ns2_x0[W_cols + Z_cols], mu1_wzx1, binary=False)
                df.loc[ts, 'mu2_wde'] = pred(df_ts[W_cols + Z_cols], mu1_wzx1, binary=False)
                if Y_binary:
                    df_ns2_x0['mu2_wde'] = np.clip(df_ns2_x0['mu2_wde'], 0, 1)
                    df.loc[ts, 'mu2_wde'] = np.clip(df.loc[ts, 'mu2_wde'], 0, 1)

                mu2_zx0 = fit(df_ns2_x0[Z_cols], df_ns2_x0['mu2_wde'],
                              binary=False, **hp_dict['mu2_zx0'])
                df.loc[ts, 'mu1_wde'] = pred(df_ts[Z_cols], mu2_zx0, binary=False)
                if Y_binary:
                    df.loc[ts, 'mu1_wde'] = np.clip(df.loc[ts, 'mu1_wde'], 0, 1)
            
            # check df prospensity scores are not too small after cross-fitting
            for model in model_list:
                if model.startswith('px'):
                    ps = df[model].astype(float)
                    print(f'{model}: min={ps.min():.4f}, mean={ps.mean():.4f}, max={ps.max():.4f}')
                    print(ps.quantile([0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 0.999]))
            # Compute effects from the cross-fitted nuisance estimates
            results = compute_effects_from_df(df, X, Y, Y_binary, W_cols)
            results_df = pd.concat([results_df, results], ignore_index=True)

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Causal effect estimation for resume LLM scores')
    parser.add_argument('--score_path', type=str, help='Path to the score CSV file')
    parser.add_argument('--X', type=str, choices=['sex', 'race'], default='sex',
                        help='Variable to treat as X (default: sex)')
    parser.add_argument('--mode', type=str, default='txt_llm_scores',
                        choices=['txt_llm_scores', 'txt_scores'],
                        help='Score mode (default: txt_llm_scores)')
    parser.add_argument('--x0', type=str, choices=['Female', 'Male', 'White', 'Non-White'], default='Female',
                        help='Control group for X (default: Female, use White/Non-White for race)')
    parser.add_argument('--resume_format', type=str,
                        choices=['no_demographics', 'with_demographics_img'], default='no_demographics',
                        help="Resume format: 'no_demographics' for text resumes and photo-free resume "
                             "images, 'with_demographics_img' for resume images carrying a profile photo "
                             "(default: no_demographics)")
    parser.add_argument('--with_skill', type=int, choices=[0, 1], default=0,
                        help='Whether resumes and job descriptions include skill section (default: 0)')
    parser.add_argument('--bootstraps', type=int, default=100,
                        help='Number of bootstrap iterations (default: 100)')
    parser.add_argument('--K', type=int, default=5,
                        help='Number of folds for cross-fitting (default: 5)')
    parser.add_argument('--clip', type=float, default=1e-4,
                        help='Clipping value for propensity scores (default: 1e-4)')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--with_state_region', action='store_true')
    parser.add_argument('--with_state_geo', action='store_true')
    parser.add_argument('--with_new_name_group', action='store_true')
    args = parser.parse_args()

    if args.with_state_region and args.with_state_geo:
        raise ValueError('Cannot use both with_state_region and with_state_geo at the same time')

    os.makedirs('hparams', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load and preprocess data
    print('Loading data...')
    print(f"Score path: {args.score_path}\nX={args.X}\n")
    df_analysis_dict = load_data(mode=args.mode, score_path=args.score_path, X0=args.x0, X_attr=args.X,) # ("with_demographics_age_race_sex", 1, "v1")

    vars_ = load_vars(X_attr=args.X, with_state_region=args.with_state_region, with_new_name_group=args.with_new_name_group, with_state_geo=args.with_state_geo, )
    X = vars_['X']
    Y = 'score'
    W_cols = vars_['W']
    V_cols = vars_['V']
    Z_cols = vars_['Z']
    Y_binary = False

    for df_key, df_analysis in df_analysis_dict.items():
        if df_key[0] != args.resume_format:
            continue
        if (df_key[1] == 1 and df_key[2] == 'v2') or (df_key[1] == 0 and df_key[2] == 'v1'):
            continue
        if args.with_skill != df_key[1]:
            continue
        print(f"CONFIG: {df_key}")
        if args.test_mode:
            continue  # Skip actual estimation in test mode
        # Encode categorical columns (names, state) to integers for XGBoost
        cat_cols = [c for c in W_cols + V_cols + Z_cols if df_analysis[c].dtype == object]
        df_analysis = encode_categoricals(df_analysis, cat_cols)
        # Estimation settings
        bootstraps = args.bootstraps
        K = args.K
        clip = args.clip
        # dataset = 'resume_llm'
        path_name = Path(args.score_path).stem

        save_path = f'results/{path_name}_{df_key[0]}_X={args.X}_x0{args.x0}_withskill{df_key[1]}_jobinfo{df_key[2]}_B={bootstraps}_K={K}_C={clip}.csv'
        if args.with_state_region:
            save_path = save_path.replace('.csv', '_with_state_region.csv')
        if args.with_state_geo:
            save_path = save_path.replace('.csv', '_with_state_geo.csv')
        if args.with_new_name_group:
            save_path = save_path.replace('.csv', '_with_new_name_group.csv')
        if os.path.exists(save_path):
            print(f'Skipping existing results at {save_path}')
            continue
        save_path = save_path.replace('processed_job_data_0102_with_exp_pred_with_names_with_resumes_datafull_', '')
            

        print(f'Running estimation: {path_name}, Y={Y}, bootstraps={bootstraps}, K={K}, clip={clip}')
        print(f'  X={X}, Z={Z_cols}, W={W_cols}, V={V_cols}')
        print(f'  N={len(df_analysis)}')
        
        hparam_path = f'hparams/{path_name}_{df_key[0]}_X={args.X}_x0{args.x0}_withskill{df_key[1]}_jobinfo{df_key[2]}.json'
        hparam_path = hparam_path.replace('processed_job_data_0102_with_exp_pred_with_names_with_resumes_datafull_', '')
        if args.with_state_region:
            hparam_path = hparam_path.replace('.json', '_with_state_region.json')
        if args.with_state_geo:
            hparam_path = hparam_path.replace('.json', '_with_state_geo.json')
        if args.with_new_name_group:
            hparam_path = hparam_path.replace('.json', '_with_new_name_group.json')
        results_df = estimate_effects(
            None, df_analysis, X, Y, W_cols, V_cols, Z_cols,
            Y_binary, [len(df_analysis)],
            bootstraps=bootstraps, K=K, clip=clip,
            hparam_path=hparam_path,
        )

        results_df.to_csv(save_path, index=False)
        print(f'Results saved to {save_path}')

        # Print summary
        print('\n' + '=' * 60)
        print('RESULTS SUMMARY (mean over bootstrap iterations)')
        print('=' * 60)
        for col in ['Estimated_TE', 'Estimated_TE_sn', 'Estimated_NDE', 'Estimated_NDE_sn',
                    'Estimated_NIE', 'Estimated_NIE_sn', 'Estimated_VDE1', 'Estimated_VDE_sn1',
                    'Estimated_WDE1', 'Estimated_WDE_sn1', 'Estimated_VDE2', 'Estimated_VDE_sn2',
                    'Estimated_WDE2', 'Estimated_WDE_sn2',
                    'Estimated_NDE*', 'Estimated_NDE*_sn',
                    'Estimated_NIE*', 'Estimated_NIE*_sn']:
            vals = results_df[col]
            print(f'  {col:35s}: {vals.mean():+.4f}  (STD={vals.std():.4f})')
