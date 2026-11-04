#!/usr/bin/env python3
"""iGCT CSF metabolomics: auditable eight-classifier analysis.

See README.md for the data contract and provenance of this implementation.
Reported manuscript results are used ONLY by the separate audit command.
Training never reads reference AUCs or changes settings to reproduce them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from functools import partial
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, LeaveOneOut, ParameterGrid, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

VERSION = "2.0.0"
MODELS = ("SVM", "KNN", "RandomForest", "LogisticRegression", "AdaBoost",
          "ExtraTrees", "GaussianNB", "MLP")
PANEL = ["m/z_174.03926", "m/z_139.04981"]
MARKERS = ["serum_afp_ng_ml", "serum_bhcg_iu_l", "csf_afp_ng_ml", "csf_bhcg_iu_l"]
META = {"sample_id", "patient_id", "label", "center", "subtype", "batch",
        "marker_negative", *MARKERS}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "openpyxl", "joblib"]
CI_NOTE = ("Single-ROC DeLong placement-value variance, normal 95% interval clipped to [0,1]. "
           "For cross-validated predictions this is descriptive: overlapping training sets "
           "and prior feature/model selection are not accounted for.")


def fail(message):
    raise ValueError(message)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(value, path):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def id_hashes(values):
    return [hashlib.sha256(str(v).encode()).hexdigest() for v in values]


def environment():
    return {"script_version": VERSION, "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in PACKAGES}}


def output_dir(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        fail(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def table(path):
    path = Path(path)
    if not path.is_file():
        fail(f"Input file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        raw = pd.read_excel(path, header=None, dtype=object, keep_default_na=False)
    else:
        raw = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",",
                          header=None, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if raw.empty:
        fail(f"Empty input: {path}")
    names = [str(v).strip() for v in raw.iloc[0]]
    if len(set(names)) != len(names) or any(not v for v in names):
        fail(f"Empty or duplicate column names: {path}")
    data = raw.iloc[1:].reset_index(drop=True)
    data.columns = names
    return data.replace(r"^\s*$", np.nan, regex=True)


def numeric(data, columns, context):
    raw = data.loc[:, columns]
    result = raw.apply(pd.to_numeric, errors="coerce").astype(float)
    if (result.isna() & raw.notna()).any().any() or np.isinf(result.to_numpy()).any():
        fail(f"Non-numeric or infinite value in {context}; use empty cells for missing values.")
    return result


def cohort(path, labels=True):
    data = table(path)
    required = {"sample_id", "patient_id"} | ({"label"} if labels else set())
    if not required.issubset(data) or data.empty:
        fail(f"{path} requires participant-level abundances and {sorted(required)}; a feature list is insufficient.")
    for col in ("sample_id", "patient_id"):
        if data[col].isna().any():
            fail(f"Missing {col} in {path}")
        data[col] = data[col].astype(str).str.strip()
        if data[col].eq("").any() or data[col].duplicated().any():
            fail(f"{col} must be nonempty and unique. Aggregate technical replicates before analysis.")
    if "label" in data:
        y = numeric(data, ["label"], "outcome labels")["label"]
        if y.isna().any() or not set(y).issubset({0, 1}):
            fail("label must be complete: 0 = non-tumor control; 1 = iGCT. Omit label for unlabeled prediction.")
        data["label"] = y.astype(int)
    features = [c for c in data if c not in META]
    if not features:
        fail("No abundance columns found.")
    x = numeric(data, features, str(path))
    if (x < 0).any().any():
        fail("Supply nonnegative abundance before cohort-level centering/scaling; negative inputs are rejected.")
    return data[[c for c in data if c in META]].copy(), x


def check_overlap(first, second):
    for key in ("patient_id", "sample_id"):
        if set(first[key]) & set(second[key]):
            fail(f"Development cohorts overlap in {key}. Use globally unique IDs; do not relabel duplicates.")


def feature_audit(path, out):
    lists = {}
    with pd.ExcelFile(path) as book:
        for sheet in book.sheet_names:
            data = pd.read_excel(book, sheet_name=sheet, dtype=str)
            if "Feature" not in data:
                fail(f"Worksheet {sheet} has no Feature column.")
            lists[sheet] = sorted({v.strip() for v in data.Feature.dropna() if v.strip()})
    names = sorted(set().union(*map(set, lists.values())))
    membership = pd.DataFrame({"feature_id": names})
    for sheet, ids in lists.items():
        membership[sheet] = membership.feature_id.isin(ids).astype(int)
    membership.to_csv(out / "feature_list_membership.csv", index=False)
    result = {"sha256": sha(path), "sheet_counts": {k: len(v) for k, v in lists.items()},
              "union_count": len(names), "manuscript_ids_present": {p: p in names for p in PANEL},
              "role": "audit_only; never restricts training candidates",
              "interpretation": "Model-list membership does not establish elevation, chemical identity, or fold stability."}
    dump(result, out / "feature_list_audit.json")
    return result


def require_keys(obj, keys, context):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        fail(f"{context} must contain exactly these keys: {sorted(keys)}")


def validate_prep(prep):
    require_keys(prep, {"transform", "missing_values"}, "preprocessing")
    if prep["transform"] not in {"none", "log2"}:
        fail("Set the actual preprocessing transform to none or log2; it cannot be inferred from the reported AUC.")
    if prep["missing_values"] not in {"error", "median"}:
        fail("Specify actual missing-value handling: error (complete inputs) or median (fitted on training only).")


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "analysis_kind", "parameter_provenance", "mode", "discovery", "internal",
                "discovery_features", "internal_features", "internal_feature_map", "candidate_workbook",
                "qc_prefiltered", "panel_features", "preprocessing", "cv", "models", "ranking_metric",
                "lock", "selection", "subgroup", "n_jobs"}
    require_keys(cfg, required, "Configuration")
    if cfg["schema_version"] != 2:
        fail("This release requires schema_version = 2.")
    if cfg["analysis_kind"] not in {"reanalysis", "original_reproduction", "software_test"}:
        fail("Set analysis_kind to reanalysis, original_reproduction, or software_test.")
    if not isinstance(cfg["parameter_provenance"], str) or not cfg["parameter_provenance"].strip():
        fail("parameter_provenance must identify the source of the actual settings, or explicitly describe a reanalysis.")
    if cfg["mode"] not in {"fixed_panel", "recorded_selection"}:
        fail("mode must be fixed_panel or recorded_selection.")
    if type(cfg["qc_prefiltered"]) is not bool:
        fail("qc_prefiltered must explicitly state whether the documented QC filters were applied.")
    if cfg["panel_features"] != PANEL:
        fail("This manuscript-specific resource uses the documented ordered NAA/urocanic-acid feature IDs. Map verified input IDs explicitly.")
    validate_prep(cfg["preprocessing"])
    cv = cfg["cv"]
    require_keys(cv, {"outer_folds", "inner_folds", "shuffle", "seed", "outer_folds_file"}, "cv")
    if cv["outer_folds"] != 5 or type(cv["inner_folds"]) is not int or cv["inner_folds"] < 2:
        fail("Specify five outer folds and the actual inner-fold count (integer >= 2).")
    if type(cv["shuffle"]) is not bool or type(cv["seed"]) is not int or not 0 <= cv["seed"] < 2**32 - 100:
        fail("Specify the actual shuffle setting and integer seed.")
    require_keys(cfg["models"], MODELS, "models")
    for name, spec in cfg["models"].items():
        require_keys(spec, {"params", "grid"}, name)
        if not isinstance(spec["params"], dict) or spec["grid"] is None:
            fail(f"Provide the actual base parameters and search grid for {name}.")
        grids = [spec["grid"]] if isinstance(spec["grid"], dict) else spec["grid"]
        if not isinstance(grids, list) or not grids or any(not isinstance(g, dict) for g in grids):
            fail(f"Invalid parameter grid for {name}.")
        for grid in grids:
            if any(not key.startswith("model__") for key in grid):
                fail("Only model__ parameters can be searched; preprocessing is explicitly fixed in the configuration.")
        list(ParameterGrid(spec["grid"]))
    if cfg["ranking_metric"] not in {"pooled_oof_auc", "mean_fold_auc"}:
        fail("Specify the actual classifier ranking metric: pooled_oof_auc or mean_fold_auc.")
    require_keys(cfg["lock"], {"model", "svm_score", "threshold", "threshold_provenance"}, "lock")
    lock = cfg["lock"]
    if lock["model"] != "SVM" or lock["svm_score"] not in {"decision_function", "probability"}:
        fail("The manuscript locks SVM. Specify its actual score type.")
    if isinstance(lock["threshold"], bool) or not isinstance(lock["threshold"], (int, float)) or not np.isfinite(lock["threshold"]):
        fail("Provide the actual pre-external SVM classification threshold; 0.273 is a separate subgroup-LR threshold.")
    if lock["svm_score"] == "probability" and not 0 <= lock["threshold"] <= 1:
        fail("A probability threshold must be between 0 and 1.")
    if not isinstance(lock["threshold_provenance"], str) or not lock["threshold_provenance"].strip():
        fail("Record where and when the locked threshold was established.")
    if type(cfg["n_jobs"]) is not int or cfg["n_jobs"] != 1:
        fail("This verified release requires n_jobs=1 so convergence failures are handled consistently in every fit.")
    require_keys(cfg["selection"], {"retained_file", "folds_file", "algorithm_description"}, "selection")
    require_keys(cfg["subgroup"], {"preprocessing", "lr_params", "parameter_provenance", "reported_probability_cutoff"}, "subgroup")
    if cfg["subgroup"]["reported_probability_cutoff"] != 0.273:
        fail("The supplied manuscript reports 0.273. This reference operating point is not optimized by the script.")
    return cfg


def config_path(config_file, value):
    return (Path(config_file).resolve().parent / value).resolve() if value else None


def prep_steps(prep):
    validate_prep(prep)
    steps = []
    # Order is explicit: training-median imputation, then optional log2, then Z-score.
    if prep["missing_values"] == "median":
        steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
    if prep["transform"] == "log2":
        steps.append(("log2", FunctionTransformer(np.log2, validate=False)))
    return steps + [("scale", StandardScaler())]


def check_x(x, prep, context, training=True):
    if x.empty or not np.isfinite(x.to_numpy()[~x.isna().to_numpy()]).all():
        fail(f"Empty or non-finite data in {context}.")
    if prep["missing_values"] == "error" and x.isna().any().any():
        fail(f"Missing abundance in {context}; the configured analysis requires complete inputs.")
    if training and x.isna().all().any():
        fail(f"All-missing training feature in {context}; held-out values cannot determine imputation.")
    if prep["transform"] == "log2" and (x <= 0).any().any():
        fail(f"log2 requires strictly positive observed values in {context}; no pseudocount is added.")


def generated_splits(y, n, cv, offset=0):
    if set(y) != {0, 1} or min(np.bincount(y)) < n:
        fail(f"Stratified {n}-fold cross-validation requires at least {n} participants per class.")
    splitter = StratifiedKFold(n_splits=n, shuffle=cv["shuffle"],
                              random_state=cv["seed"] + offset if cv["shuffle"] else None)
    return list(splitter.split(np.zeros(len(y)), y))


def recorded_splits(meta, path):
    folds = table(path)
    if set(folds) != {"sample_id", "fold"} or folds.sample_id.duplicated().any():
        fail("Fold file must contain unique sample_id and fold columns.")
    if set(folds.sample_id) != set(meta.sample_id):
        fail("Fold assignments must cover exactly the discovery samples.")
    ids = pd.to_numeric(folds.set_index("sample_id").loc[meta.sample_id, "fold"], errors="coerce").to_numpy()
    if set(ids) != {1, 2, 3, 4, 5}:
        fail("Recorded folds must be integer 1 through 5.")
    y = meta.label.to_numpy()
    result = []
    for fold in range(1, 6):
        train, test = np.flatnonzero(ids != fold), np.flatnonzero(ids == fold)
        if set(y[train]) != {0, 1} or set(y[test]) != {0, 1}:
            fail("Each recorded training and held-out partition must contain both outcome classes.")
        result.append((train, test))
    return result


def auc_ci(y, score):
    y, score = np.asarray(y), np.asarray(score, float)
    if set(y) != {0, 1} or not np.isfinite(score).all() or len(y) != len(score):
        fail("AUC requires two outcome classes and complete finite scores.")
    a, b = score[y == 1], score[y == 0]
    values = (a[:, None] > b).astype(float) + 0.5 * (a[:, None] == b)
    auc = float(values.mean())
    if min(len(a), len(b)) < 2:
        return auc, None, None
    variance = np.var(values.mean(axis=1), ddof=1) / len(a) + np.var(values.mean(axis=0), ddof=1) / len(b)
    delta = norm.ppf(0.975) * np.sqrt(max(float(variance), 0))
    return auc, max(0.0, auc - delta), min(1.0, auc + delta)


def metrics(y, score, threshold=None):
    y, score = np.asarray(y), np.asarray(score, float)
    auc, lo, hi = auc_ci(y, score)
    result = {"n": len(y), "n_case": int(sum(y == 1)), "n_control": int(sum(y == 0)),
              "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi}
    if threshold is not None:
        tn, fp, fn, tp = confusion_matrix(y, score >= threshold, labels=[0, 1]).ravel()
        result.update(threshold=float(threshold), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                      accuracy=(tp + tn) / len(y), youden=tp / (tp + fn) - fp / (tn + fp),
                      tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))
    return clean_json(result)


def youden(y, score):
    """Exact rational Youden comparison; among tied finite thresholds, choose the highest."""
    y, score = np.asarray(y), np.asarray(score, float)
    fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
    n1, n0 = sum(y == 1), sum(y == 0)
    objective = np.rint(tpr * n1).astype(int) * n0 - np.rint(fpr * n0).astype(int) * n1
    finite = np.isfinite(thresholds)
    # A finite observed threshold is used; an all-negative classifier at +infinity is not an assay cutoff.
    threshold = thresholds[finite][np.argmax(objective[finite])]
    return float(threshold)


def roc_plot(y, series, destination, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for name, score in series.items():
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name}: AUC {roc_auc_score(y, score):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="0.7", lw=1)
    ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.02), title=title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(destination) + ".pdf")
    fig.savefig(str(destination) + ".png", dpi=300)
    plt.close(fig)


def estimator(name, spec):
    classes = {"SVM": SVC, "KNN": KNeighborsClassifier, "RandomForest": RandomForestClassifier,
               "LogisticRegression": LogisticRegression, "AdaBoost": AdaBoostClassifier,
               "ExtraTrees": ExtraTreesClassifier, "GaussianNB": GaussianNB, "MLP": MLPClassifier}
    params = dict(spec["params"])
    if name == "AdaBoost":
        stump = params.pop("base_tree_params", None)
        if not isinstance(stump, dict):
            fail("AdaBoost requires explicit base_tree_params. This tested version implements SAMME, not SAMME.R.")
        params["estimator"] = DecisionTreeClassifier(**stump)
    return classes[name](**params)


def model_score(fitted, x, svm_score):
    if isinstance(fitted.named_steps["model"], SVC) and svm_score == "decision_function":
        return np.asarray(fitted.decision_function(x), float)
    return np.asarray(fitted.predict_proba(x)[:, list(fitted.classes_).index(1)], float)


def scoring(fitted, x, y, svm_score):
    return roc_auc_score(y, model_score(fitted, x, svm_score))


def fit_checked(model, x, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model, sorted({str(w.message) for w in caught})


def fit_search(x, y, name, cfg, splits):
    check_x(x, cfg["preprocessing"], f"{name} full training")
    for i, (train, _) in enumerate(splits, 1):
        check_x(x.iloc[train], cfg["preprocessing"], f"{name} inner training fold {i}")
    model = estimator(name, cfg["models"][name])
    pipe = Pipeline(prep_steps(cfg["preprocessing"]) + [("model", model)])
    if name == "SVM" and cfg["lock"]["svm_score"] == "probability":
        if not model.get_params()["probability"]:
            fail("SVM probability scoring requires probability=True before fitting.")
    grid = cfg["models"][name]["grid"]
    for pars in ParameterGrid(grid):
        candidate = clone(pipe).set_params(**pars)
        if name == "SVM" and cfg["lock"]["svm_score"] == "probability" and not candidate.named_steps["model"].probability:
            fail("Every SVM candidate must use probability=True for probability scoring.")
    search = GridSearchCV(pipe, grid, cv=splits, scoring=partial(scoring, svm_score=cfg["lock"]["svm_score"]),
                          n_jobs=cfg["n_jobs"], refit=True, error_score="raise", return_train_score=False)
    return fit_checked(search, x, y)


def parameter_snapshot(fitted):
    values = fitted.named_steps["model"].get_params(deep=True)
    values.pop("estimator", None)  # Nested tree parameters are retained below.
    result = {"model_parameters": values}
    scaler = fitted.named_steps["scale"]
    result["scaler"] = {"mean": scaler.mean_, "scale": scaler.scale_, "var": scaler.var_}
    if "imputer" in fitted.named_steps:
        result["imputer_training_medians"] = fitted.named_steps["imputer"].statistics_
    model = fitted.named_steps["model"]
    if isinstance(model, LogisticRegression):
        result.update(coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_)
    if isinstance(model, SVC):
        result.update(fitted_gamma=model._gamma, n_support=model.n_support_, intercept=model.intercept_)
    return clean_json(result)


def benchmark(x, y, meta, cfg, splits, out):
    summary, details, inner_rows, pred = [], [], [], meta[["sample_id", "patient_id", "label"]].copy()
    pred["outer_fold"] = 0
    for fold, (train, test) in enumerate(splits, 1):
        pred.loc[test, "outer_fold"] = fold
        inner = generated_splits(y[train], cfg["cv"]["inner_folds"], cfg["cv"], fold)
        for j, (_, holdout) in enumerate(inner, 1):
            inner_rows.extend({"outer_fold": fold, "inner_fold": j, "sample_id": meta.iloc[train[k]].sample_id} for k in holdout)
    pd.DataFrame(inner_rows).to_csv(out / "inner_fold_assignments.csv", index=False)
    for name in MODELS:
        print(f"Evaluating {name} ({MODELS.index(name) + 1}/8)", flush=True)
        oof = np.full(len(y), np.nan)
        fold_auc = []
        for fold, (train, test) in enumerate(splits, 1):
            inner = generated_splits(y[train], cfg["cv"]["inner_folds"], cfg["cv"], fold)
            search, notes = fit_search(x.iloc[train], y[train], name, cfg, inner)
            oof[test] = model_score(search.best_estimator_, x.iloc[test], cfg["lock"]["svm_score"])
            auc = float(roc_auc_score(y[test], oof[test]))
            fold_auc.append(auc)
            details.append({"model": name, "fold": fold, "n_train": len(train), "n_test": len(test),
                            "auc": auc, "best_parameters": search.best_params_,
                            "fitted_training_parameters": parameter_snapshot(search.best_estimator_), "warnings": notes})
            pd.DataFrame(search.cv_results_).to_csv(out / f"grid_{name}_fold{fold}.csv", index=False)
        row = metrics(y, oof)
        row.update(model=name, pooled_oof_auc=row["auc"], mean_fold_auc=float(np.mean(fold_auc)),
                   sd_fold_auc=float(np.std(fold_auc, ddof=1)),
                   score_type=cfg["lock"]["svm_score"] if name == "SVM" else "probability")
        summary.append(row)
        pred[name] = oof
    report = pd.DataFrame(summary).sort_values(cfg["ranking_metric"], ascending=False, kind="stable")
    report.to_csv(out / "eight_model_metrics.csv", index=False)
    pred.to_csv(out / "discovery_oof_predictions.csv", index=False)
    dump(details, out / "fold_parameters.json")
    roc_plot(y, {name: pred[name] for name in MODELS}, out / "discovery_oof_roc", "Discovery: selected panel, pooled OOF scores")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(np.arange(8), report.mean_fold_auc, yerr=report.sd_fold_auc, capsize=3, color="#607c94")
    ax.set_xticks(np.arange(8), report.model, rotation=30, ha="right")
    ax.set(ylabel="Fold AUC (mean and SD)", ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(out / "eight_model_fold_auc.pdf")
    plt.close(fig)
    return report


def annotation(path, discovery, qc_prefiltered):
    data = table(path)
    needed = ["feature_id", "mz", "rt_min"] + (["vip", "peak_score"] if discovery else [])
    if not set(needed).issubset(data) or data.empty:
        fail(f"Feature metadata requires {needed}: {path}")
    if data.feature_id.isna().any():
        fail("Feature IDs cannot be missing.")
    data["feature_id"] = data.feature_id.astype(str).str.strip()
    if data.feature_id.eq("").any() or data.feature_id.duplicated().any():
        fail("Feature IDs must be unique and nonempty.")
    numbers = numeric(data, needed[1:], str(path))
    if numbers.isna().any().any() or (numbers.mz <= 0).any() or (numbers.rt_min < 0).any():
        fail("Missing or invalid m/z, RT, VIP, or peak-quality score.")
    for col in numbers:
        data[col] = numbers[col]
    if {"qc_detection_rate", "qc_rsd"}.issubset(data):
        qc = numeric(data, ["qc_detection_rate", "qc_rsd"], "QC")
        if qc.isna().any().any() or not qc.qc_detection_rate.between(0, 1).all() or (qc.qc_rsd < 0).any():
            fail("QC detection rate and RSD must be complete proportions, not percentages.")
        data["qc_pass"] = (qc.qc_detection_rate >= 0.8) & (qc.qc_rsd <= 0.3)
    elif qc_prefiltered:
        data["qc_pass"] = True
    else:
        fail("Provide pooled-QC metrics or confirm that the documented QC filters were actually applied.")
    return data.set_index("feature_id", drop=False)


def bh(p):
    p = np.asarray(p, float)
    result = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    order = np.argsort(p[ok])
    n = int(ok.sum())
    if n:
        adjusted = np.minimum.accumulate((p[ok][order] * n / np.arange(1, n + 1))[::-1])[::-1]
        restored = np.empty(n)
        restored[order] = np.minimum(adjusted, 1)
        result[ok] = restored
    return result


def mean_ratio(case, control):
    a, b = np.asarray(case, float), np.asarray(control, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(a.mean() / b.mean()) if len(a) and len(b) and b.mean() > 0 else np.nan


def screen(x, y, ann, out):
    if set(x) - set(ann.index):
        fail("Every discovery feature must have actual QC, OPLS-DA VIP, and peak-quality metadata.")
    rows = []
    for f in x:
        a, b = x.loc[y == 1, f].dropna(), x.loc[y == 0, f].dropna()
        p = float(mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").pvalue) if min(len(a), len(b)) >= 2 else np.nan
        fc, info = mean_ratio(a, b), ann.loc[f]
        selected = bool(np.isfinite(p) and p < 0.05 and np.isfinite(fc) and fc > 2 and
                        info.vip > 1 and info.peak_score >= 3 and info.qc_pass)
        rows.append({"feature_id": f, "n_case_observed": len(a), "n_control_observed": len(b),
                     "p_value": p, "fold_change_mean_ratio": fc, "vip": info.vip,
                     "peak_score": info.peak_score, "qc_pass": bool(info.qc_pass), "selected": selected})
    frame = pd.DataFrame(rows)
    frame["q_BH_within_supplied_universe"] = bh(frame.p_value)
    frame.to_csv(out / "initial_candidate_screen.csv", index=False)
    return frame.loc[frame.selected, "feature_id"].tolist()


def mapping_file(path, left, right):
    data = table(path)
    if set(data) != {left, right} or data.empty or data.isna().any().any():
        fail(f"Mapping requires complete {left},{right} columns.")
    for col in (left, right):
        data[col] = data[col].astype(str).str.strip()
        if data[col].eq("").any() or data[col].duplicated().any():
            fail("Feature mappings must be nonempty and one-to-one.")
    return dict(zip(data[left], data[right]))


def replicate(names, da, ia, x, y, map_path, out):
    explicit = mapping_file(map_path, "discovery_feature_id", "internal_feature_id") if map_path else {}
    valid = ia.loc[ia.qc_pass & ia.index.isin(x.columns)]
    rows, used, matched = [], set(), {}
    for f in names:
        dm = (valid.mz - da.loc[f, "mz"]).abs()
        dr = (valid.rt_min - da.loc[f, "rt_min"]).abs()
        possible = valid.index[(dm <= 0.03) & (dr <= 0.3)].tolist()
        target = explicit.get(f)
        if target is not None and target not in possible:
            fail(f"Explicit match {f}->{target} fails QC or m/z/RT tolerance.")
        if target is None:
            if f in possible:
                target = f
            elif len(possible) == 1:
                target = possible[0]
            elif len(possible) > 1:
                fail(f"Ambiguous match for {f}; supply a verified mapping.")
        fc = np.nan
        if target is not None:
            if target in used:
                fail("Multiple discovery features match the same internal feature.")
            used.add(target)
            matched[f] = target
            fc = mean_ratio(x.loc[y == 1, target], x.loc[y == 0, target])
        rows.append({"feature_id": f, "internal_feature_id": target, "internal_fold_change": fc,
                     "delta_mz": dm[target] if target else None, "delta_rt_min": dr[target] if target else None,
                     "reproduced": bool(np.isfinite(fc) and fc > 1)})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "internal_reproducibility.csv", index=False)
    return frame.loc[frame.reproduced, "feature_id"].tolist(), matched


def recorded_selection(names, meta, selection, config_file, out):
    if not isinstance(selection["algorithm_description"], str) or not selection["algorithm_description"].strip():
        fail("Supply the original within-fold SVM retention rule. SVM-RFE is not implied by the manuscript.")
    if not selection["retained_file"] or not selection["folds_file"]:
        fail("Supply original fold-level retention decisions and discovery fold assignments.")
    recorded_splits(meta, config_path(config_file, selection["folds_file"]))
    data = table(config_path(config_file, selection["retained_file"]))
    if set(data) != {"feature_id", "fold", "retained"} or data.duplicated(["feature_id", "fold"]).any():
        fail("Retention log requires one row per feature/fold: feature_id,fold,retained.")
    data[["fold", "retained"]] = numeric(data, ["fold", "retained"], "retention log")
    if data.isna().any().any() or set(data.feature_id) != set(names) or set(data.fold) != {1, 2, 3, 4, 5} or not set(data.retained).issubset({0, 1}):
        fail("Retention log must cover exactly the reproduced features, all five folds, and binary retention decisions.")
    if not data.groupby("feature_id").size().eq(5).all():
        fail("Every reproduced feature requires five explicit retention decisions, including zeros.")
    result = data.pivot(index="feature_id", columns="fold", values="retained").astype(int)
    result.columns = [f"retained_fold_{int(c)}" for c in result.columns]
    result["stability_score_0_to_5"] = result.sum(axis=1)
    result = result.sort_values("stability_score_0_to_5", ascending=False, kind="stable")
    result.to_csv(out / "recorded_stability_scores.csv")
    if len(result) < 2 or result.iloc[1, -1] == 0:
        fail("The recorded logs do not identify two informative features.")
    if len(result) > 2 and result.iloc[1, -1] == result.iloc[2, -1]:
        fail("Tie at the second selection position; retrieve the original tie-breaking rule instead of substituting the manuscript panel.")
    return result.index[:2].tolist()


def train(config_file, out):
    cfg = read_config(config_file)
    path = partial(config_path, config_file)
    ds, dx = cohort(path(cfg["discovery"]))
    ins, ix = cohort(path(cfg["internal"]))
    check_overlap(ds, ins)
    if set(ins.label) != {0, 1}:
        fail("The internal reproducibility set must include cases and controls.")
    y = ds.label.to_numpy(int)
    audit = feature_audit(path(cfg["candidate_workbook"]), out) if cfg["candidate_workbook"] else None
    initial, reproduced, imap = None, None, {}
    if cfg["mode"] == "recorded_selection":
        da = annotation(path(cfg["discovery_features"]), True, cfg["qc_prefiltered"])
        ia = annotation(path(cfg["internal_features"]), False, cfg["qc_prefiltered"])
        initial = screen(dx, y, da, out)
        if len(initial) < 2:
            fail("Fewer than two screened candidates. No feature is inserted to match the manuscript.")
        reproduced, imap = replicate(initial, da, ia, ix, ins.label.to_numpy(int), path(cfg["internal_feature_map"]), out)
        selected = recorded_selection(reproduced, ds, cfg["selection"], config_file, out)
        if set(selected) != set(PANEL):
            dump({"selected_features": selected, "matches_manuscript_panel": False}, out / "selection_discrepancy.json")
            fail("Recorded stability results do not select the manuscript panel; inspect selection_discrepancy.json.")
    elif not cfg["qc_prefiltered"]:
        fail("Fixed-panel benchmarking requires explicit confirmation of the documented upstream QC.")
    if set(PANEL) - set(dx):
        fail("The exact manuscript panel is absent from discovery abundances. Do not substitute nearby masses.")
    x = dx[PANEL]
    check_x(x, cfg["preprocessing"], "discovery")
    splits = recorded_splits(ds, path(cfg["cv"]["outer_folds_file"])) if cfg["cv"]["outer_folds_file"] else generated_splits(y, 5, cfg["cv"])
    summary = benchmark(x, y, ds, cfg, splits, out)
    final_splits = generated_splits(y, cfg["cv"]["inner_folds"], cfg["cv"], 100)
    final, notes = fit_search(x, y, "SVM", cfg, final_splits)
    final_assignments = []
    for fold, (_, test) in enumerate(final_splits, 1):
        final_assignments.extend({"sample_id": ds.iloc[i].sample_id, "fold": fold} for i in test)
    pd.DataFrame(final_assignments).to_csv(out / "final_fit_cv_assignments.csv", index=False)
    pd.DataFrame(final.cv_results_).to_csv(out / "final_svm_grid.csv", index=False)
    max_metric = summary[cfg["ranking_metric"]].max()
    top = summary.loc[np.isclose(summary[cfg["ranking_metric"]], max_metric, rtol=0, atol=1e-12), "model"].tolist()
    source_paths = [Path(config_file).resolve(), path(cfg["discovery"]), path(cfg["internal"])]
    for key in ("discovery_features", "internal_features", "internal_feature_map", "candidate_workbook"):
        if cfg[key]:
            source_paths.append(path(cfg[key]))
    for value in (cfg["cv"]["outer_folds_file"], cfg["selection"]["retained_file"], cfg["selection"]["folds_file"]):
        if value:
            source_paths.append(path(value))
    report = {"analysis_kind": cfg["analysis_kind"], "parameter_provenance": cfg["parameter_provenance"],
              "discovery_counts": {"case": int(sum(y == 1)), "control": int(sum(y == 0))},
              "internal_counts": {"case": int(sum(ins.label == 1)), "control": int(sum(ins.label == 0))},
              "initial_selected_count": len(initial) if initial is not None else None,
              "reproduced_count": len(reproduced) if reproduced is not None else None,
              "selection_scope": "recorded fold decisions; not a refit of the unknown selection algorithm" if initial is not None else "reported fixed panel; feature selection not re-executed",
              "panel_features": PANEL, "locked_model": "SVM", "ranking_metric": cfg["ranking_metric"],
              "top_ranked_models": top, "svm_is_top_ranked": "SVM" in top,
              "discovery_svm_oof": summary.set_index("model").loc["SVM"].to_dict(),
              "locked_threshold": cfg["lock"]["threshold"], "score_type": cfg["lock"]["svm_score"],
              "locked_best_parameters": final.best_params_, "locked_fit": parameter_snapshot(final.best_estimator_),
              "fit_warnings": notes, "feature_workbook_audit": audit, "environment": environment(),
              "source_code_sha256": sha(__file__), "input_sha256": {str(p): sha(p) for p in source_paths},
              "ci_method": CI_NOTE, "discovery_interpretation": "Conditional on preceding selection; not unbiased end-to-end CV."}
    bundle = {"version": VERSION, "environment": environment(), "pipeline": final.best_estimator_, "features": PANEL,
              "threshold": cfg["lock"]["threshold"], "score_type": cfg["lock"]["svm_score"],
              "config": cfg, "internal_feature_map": imap, "report": report,
              "discovery_patient_hashes": id_hashes(ds.patient_id), "discovery_sample_hashes": id_hashes(ds.sample_id),
              "internal_patient_hashes": id_hashes(ins.patient_id), "internal_sample_hashes": id_hashes(ins.sample_id)}
    joblib.dump(bundle, out / "locked_model.joblib")
    report["locked_model_sha256"] = sha(out / "locked_model.joblib")
    dump(report, out / "run_report.json")
    dump(cfg, out / "used_config.json")
    print(f"Locked SVM saved. Top classifier(s) by {cfg['ranking_metric']}: {', '.join(top)}.")


def load_bundle(path):
    # joblib/pickle can execute code. Load only locally created or otherwise trusted files.
    bundle = joblib.load(path)
    if bundle.get("version") != VERSION:
        fail("Model-bundle version mismatch. Use the script that created this model.")
    actual = environment()
    for name in ("numpy", "scipy", "scikit-learn", "joblib"):
        if actual["packages"][name] != bundle["environment"]["packages"][name]:
            fail(f"Saved-model dependency mismatch: {name}. Restore the recorded environment.")
    return bundle


def align(x, bundle, mapping, role):
    names = {f: f for f in bundle["features"]}
    if role == "internal" and bundle["internal_feature_map"]:
        names = {f: bundle["internal_feature_map"][f] for f in names}
    if mapping:
        names = mapping_file(mapping, "panel_feature_id", "input_feature_id")
    if set(names) != set(bundle["features"]) or set(names.values()) - set(x):
        fail("Both exact panel features and a complete one-to-one verified mapping are required.")
    aligned = pd.DataFrame({f: x[names[f]] for f in bundle["features"]})
    check_x(aligned, bundle["config"]["preprocessing"], f"{role} prediction", training=False)
    return aligned


def check_test_ids(meta, bundle, role):
    for key, part in (("patient_id", "patient"), ("sample_id", "sample")):
        ids = set(id_hashes(meta[key]))
        if ids & set(bundle[f"discovery_{part}_hashes"]):
            fail(f"Prediction data overlap discovery {key}.")
        internal = set(bundle[f"internal_{part}_hashes"])
        if role == "external" and ids & internal:
            fail(f"External data overlap the internal development set in {key}.")
        if role == "internal" and not ids.issubset(internal):
            fail("Internal prediction must use participants registered in the development record.")


def predict(model_file, data_file, mapping, role, out):
    model_hash = sha(model_file)
    bundle = load_bundle(model_file)
    meta, x = cohort(data_file, labels=False)
    check_test_ids(meta, bundle, role)
    x = align(x, bundle, mapping, role)
    scores = model_score(bundle["pipeline"], x, bundle["score_type"])
    output = meta.copy()
    output["score"] = scores
    output["predicted_label"] = (scores >= bundle["threshold"]).astype(int)
    output.to_csv(out / "fixed_model_predictions.csv", index=False)
    report = {"role": role, "analysis_kind": bundle["config"]["analysis_kind"], "refitted": False,
              "model_sha256": model_hash, "input_sha256": sha(data_file), "mapping_sha256": sha(mapping) if mapping else None,
              "score_type": bundle["score_type"], "threshold": bundle["threshold"], "features": bundle["features"],
              "metrics": None, "single_features": {}, "ci_method": CI_NOTE,
              "independence_check": "ID disjointness verified; this cannot establish absence of prior test-data use.",
              "environment": environment()}
    if "label" in meta and set(meta.label) == {0, 1}:
        y = meta.label.to_numpy(int)
        report["metrics"] = metrics(y, scores, bundle["threshold"])
        for f in x:
            observed = x[f].notna()
            if set(y[observed]) == {0, 1}:
                report["single_features"][f] = metrics(y[observed], x.loc[observed, f])
        roc_plot(y, {"Locked SVM": scores}, out / "fixed_model_roc", "External cohort" if role == "external" else "Internal reproducibility set")
    if sha(model_file) != model_hash:
        fail("The locked model changed during prediction.")
    dump(report, out / "fixed_model_report.json")


def classify_markers(meta):
    if not set(MARKERS).issubset(meta):
        fail("Supply all four marker columns (empty cells allowed). Units are AFP ng/mL and beta-hCG IU/L.")
    values = numeric(meta, MARKERS, "tumor-marker measurements")
    if (values < 0).any().any():
        fail("Tumor-marker values cannot be negative.")
    rows = []
    for _, row in values.iterrows():
        present = row.notna().to_numpy()
        positive = ((row.to_numpy() >= np.array([7, 2.6, 7, 2.6])) & present).any()
        serum_complete, csf_complete = present[:2].all(), present[2:].all()
        if positive:
            status, reason = 0, "at_least_one_observed_marker_at_or_above_cutoff"
        elif serum_complete and csf_complete:
            status, reason = 1, "both_markers_negative_in_both_fluids"
        elif serum_complete and not present[2:].any():
            status, reason = 1, "serum_only_both_negative"
        elif csf_complete and not present[:2].any():
            status, reason = 1, "csf_only_both_negative"
        else:
            status, reason = np.nan, "unclassifiable_incomplete_marker_pair" if present.any() else "unclassifiable_no_measurements"
        rows.append({"derived_marker_negative": status, "classification_reason": reason})
    result = pd.DataFrame(rows, index=meta.index)
    if "marker_negative" in meta:
        recorded = numeric(meta, ["marker_negative"], "recorded marker status")["marker_negative"]
        if not set(recorded.dropna()).issubset({0, 1}):
            fail("Optional marker_negative must be 0, 1, or empty.")
        if (recorded.notna() & (recorded != result.derived_marker_negative)).any():
            fail("Recorded marker_negative conflicts with the available measurements; resolve the clinical record.")
    return result


def subgroup(model_file, data_file, mapping, out):
    model_hash = sha(model_file)
    bundle = load_bundle(model_file)
    cfg = bundle["config"]["subgroup"]
    validate_prep(cfg["preprocessing"])
    if not isinstance(cfg["lr_params"], dict) or not cfg["parameter_provenance"]:
        fail("Provide the actual subgroup LR specification and its provenance; regularization cannot be inferred from AUC 0.940.")
    meta, raw = cohort(data_file)
    check_test_ids(meta, bundle, "external")
    # Feature alignment must not impose the main model's missing-value/log-transform rules on a separate LR analysis.
    names = {f: f for f in bundle["features"]}
    if mapping:
        names = mapping_file(mapping, "panel_feature_id", "input_feature_id")
    if set(names) != set(bundle["features"]) or set(names.values()) - set(raw):
        fail("Both panel features are required for subgroup analysis.")
    raw = pd.DataFrame({f: raw[names[f]] for f in bundle["features"]})
    marker = classify_markers(meta)
    pd.concat([meta, marker], axis=1).to_csv(out / "marker_classification.csv", index=False)
    status = marker.derived_marker_negative
    keep = (meta.label == 0) | ((meta.label == 1) & (status == 1))
    sub = meta.loc[keep].reset_index(drop=True)
    x = raw.loc[keep].reset_index(drop=True)
    y = sub.label.to_numpy(int)
    if set(y) != {0, 1} or min(np.bincount(y)) < 3:
        fail("Exploratory LR/LOOCV requires at least three cases and three controls.")
    check_x(x, cfg["preprocessing"], "subgroup")
    base = Pipeline(prep_steps(cfg["preprocessing"]) + [("model", LogisticRegression(**cfg["lr_params"]))])
    fitted, notes = fit_checked(clone(base), x, y)
    apparent = fitted.predict_proba(x)[:, 1]
    oof, folds = np.full(len(y), np.nan), []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(x), 1):
        check_x(x.iloc[train_idx], cfg["preprocessing"], f"LOOCV training fold {fold}")
        fitted_fold, warns = fit_checked(clone(base), x.iloc[train_idx], y[train_idx])
        oof[test_idx] = fitted_fold.predict_proba(x.iloc[test_idx])[:, 1]
        folds.append({"held_out_sample_id": sub.iloc[test_idx[0]].sample_id,
                      "training_fit": parameter_snapshot(fitted_fold), "warnings": warns})
    cutoff = youden(y, apparent)
    singles = {}
    for f in x:
        observed = x[f].notna()
        yy, ss = y[observed], x.loc[observed, f].to_numpy()
        if set(yy) == {0, 1}:
            singles[f] = metrics(yy, ss, youden(yy, ss))
    report = {"analysis": "Exploratory subgroup LR refit; not independent locked-SVM performance",
              "analysis_kind": bundle["config"]["analysis_kind"], "n_case": int(sum(y == 1)), "n_control": int(sum(y == 0)),
              "excluded_marker_positive_cases": int(((meta.label == 1) & (status == 0)).sum()),
              "unclassifiable_cases": int(((meta.label == 1) & status.isna()).sum()),
              "marker_thresholds": {"AFP_ng_ml": 7, "beta_hCG_iu_l": 2.6, "negative_operator": "<"},
              "preprocessing_refitted_each_LOOCV_iteration": True, "hyperparameters_fixed_before_LOOCV": True,
              "apparent_at_Youden": metrics(y, apparent, cutoff),
              "apparent_at_reported_0_273": metrics(y, apparent, cfg["reported_probability_cutoff"]),
              "reported_cutoff_is_Youden_optimal": bool(np.isclose(metrics(y, apparent, cfg["reported_probability_cutoff"])["youden"], metrics(y, apparent, cutoff)["youden"], rtol=0, atol=1e-12)),
              "LOOCV": metrics(y, oof), "single_features": singles,
              "apparent_fit": parameter_snapshot(fitted), "apparent_fit_warnings": notes,
              "parameter_provenance": cfg["parameter_provenance"], "ci_method": CI_NOTE,
              "model_sha256": model_hash, "input_sha256": sha(data_file), "mapping_sha256": sha(mapping) if mapping else None,
              "environment": environment()}
    sub["LR_apparent_probability"], sub["LR_LOOCV_probability"] = apparent, oof
    sub.to_csv(out / "subgroup_predictions.csv", index=False)
    dump(folds, out / "LOOCV_training_parameters.json")
    dump(report, out / "subgroup_exploratory_report.json")
    # These are newly fitted exploratory parameters, saved separately from the primary SVM.
    joblib.dump({"pipeline": fitted, "features": bundle["features"], "config": cfg,
                 "environment": environment(), "analysis": report["analysis"]}, out / "exploratory_lr.joblib")
    roc_plot(y, {"Refitted LR apparent": apparent, "Refitted LR LOOCV": oof}, out / "subgroup_lr_roc", "Exploratory marker-negative subgroup")
    if sha(model_file) != model_hash:
        fail("The primary model changed during exploratory analysis.")


def get_nested(value, parts):
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def audit_results(reference_file, training, external, exploratory, out):
    reference = json.loads(Path(reference_file).read_text(encoding="utf-8"))
    reports = {}
    for name, path in (("training", training), ("external", external), ("subgroup", exploratory)):
        reports[name] = json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    rows = []
    for item in reference["checks"]:
        actual = get_nested(reports.get(item["source"]), item["path"])
        expected = item["reported"]
        if actual is None:
            status = "not_assessed"
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            status = "match" if isinstance(actual, (int, float)) and abs(actual - expected) <= item.get("absolute_tolerance", 0) else "different"
        else:
            status = "match" if actual == expected else "different"
        rows.append({"check": item["name"], "reported": expected, "computed": actual, "status": status})
    findings = []
    if reports["external"] and reports["external"].get("role") != "external":
        findings.append("The supplied external report is not marked as an external cohort.")
    if reports["training"]:
        locked_hash = reports["training"].get("locked_model_sha256")
        for name in ("external", "subgroup"):
            if reports[name] and reports[name].get("model_sha256") != locked_hash:
                findings.append(f"{name} used a different locked-model artifact.")
    if reports["external"] and reports["subgroup"]:
        if reports["external"].get("input_sha256") != reports["subgroup"].get("input_sha256"):
            findings.append("External and subgroup reports did not use the same full external input file.")
    passed = bool(rows) and all(r["status"] == "match" for r in rows) and not findings
    result = {"rounded_reported_values_match": passed, "checks": rows, "provenance_findings": findings,
              "scope": "Numerical reconciliation only. A match does not verify historical settings, chemical identity, or independence.",
              "source_reports": {k: v.get("analysis_kind") if v else None for k, v in reports.items()}}
    dump(result, out / "manuscript_comparison.json")
    pd.DataFrame(rows).to_csv(out / "manuscript_comparison.csv", index=False)
    print(f"Manuscript comparison: {sum(r['status'] == 'match' for r in rows)}/{len(rows)} match; {len(findings)} provenance discrepancies.")
    return 0 if passed else 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="Audit a feature-only workbook without fitting")
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train", help="Eight-model comparison and discovery-only SVM fitting")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    for command in ("predict", "subgroup"):
        p = sub.add_parser(command)
        p.add_argument("--model", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--mapping")
        p.add_argument("--out", required=True)
        if command == "predict":
            p.add_argument("--role", choices=["external", "internal"], default="external")
    p = sub.add_parser("audit", help="Compare computed outputs with manuscript values; never tunes models")
    p.add_argument("--reference", required=True)
    p.add_argument("--training")
    p.add_argument("--external")
    p.add_argument("--subgroup")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = None
    try:
        out = output_dir(args.out)
        if args.command == "inspect":
            feature_audit(args.features, out)
        elif args.command == "train":
            train(args.config, out)
        elif args.command == "predict":
            predict(args.model, args.data, args.mapping, args.role, out)
        elif args.command == "subgroup":
            subgroup(args.model, args.data, args.mapping, out)
        else:
            return audit_results(args.reference, args.training, args.external, args.subgroup, out)
    except (ValueError, TypeError, KeyError, OSError, ConvergenceWarning) as error:
        if out is not None:
            dump({"status": "failed", "command": args.command, "error": str(error)}, out / "FAILED.json")
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
