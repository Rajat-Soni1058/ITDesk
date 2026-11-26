"""
train.py
--------
Trains category and priority classifiers on the Zenodo dataset.
Saves models to models/ and charts to outputs/.

Run after: python load_data.py
"""

import os, sys, json, joblib, warnings, shutil, tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.base import clone

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             make_scorer, f1_score)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from preprocessor import build_features, get_tfidf, META_COLS, clean_text

BASE = os.path.dirname(__file__)
MDL  = os.path.join(BASE, "models")
OUT  = os.path.join(BASE, "outputs")
os.makedirs(MDL, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COLORS     = ["#457b9d","#e63946","#f4a261","#2a9d8f","#8338ec","#06d6a0","#ffb703"]
PRI_COLORS = {"High":"#e63946","Medium":"#f4a261","Low":"#2a9d8f"}


def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels)-1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_model_comparison(results, title, path):
    names  = [r["model"] for r in results]
    scores = [r["mean"]  for r in results]
    stds   = [r["std"]   for r in results]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, scores, xerr=stds, color="#457b9d", alpha=0.85,
                   error_kw=dict(ecolor="#e63946", capsize=4))
    for bar, s in zip(bars, scores):
        ax.text(s + 0.005, bar.get_y() + bar.get_height()/2,
                f"{s:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("CV Macro F1 (mean ± std)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_class_f1(report, labels, title, colors, path):
    f1s = [report.get(l, {}).get("f1-score", 0) for l in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.5), 5))
    bars = ax.bar(labels, f1s, color=colors[:len(labels)], alpha=0.9)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def main():
    print("=" * 60)
    print("  ZENODO IT TICKET CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    data_path = os.path.join(BASE, "Data", "tickets_clean.csv")
    if not os.path.exists(data_path):
        print("\nERROR: tickets_clean.csv not found. Run load_data.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    # Remove categories with fewer than 10 tickets (too small for CV)
    min_samples = 10
    category_counts = df['category'].value_counts()
    valid_categories = category_counts[category_counts >= min_samples].index
    df = df[df['category'].isin(valid_categories)]
    print(f"Kept {df['category'].nunique()} categories after removing small classes")
    print(df['category'].value_counts())
    print(f"\n[1/6] Loaded {len(df)} tickets, {df['category'].nunique()} categories")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate_dir = os.path.join(MDL, "candidates", timestamp)
    candidate_out  = os.path.join(OUT, "candidates", timestamp)
    os.makedirs(candidate_dir, exist_ok=True)
    os.makedirs(candidate_out, exist_ok=True)

    # ── EDA plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cc = df["category"].value_counts()
    axes[0].barh(cc.index, cc.values, color=COLORS[:len(cc)], alpha=0.88)
    for i, v in enumerate(cc.values):
        axes[0].text(v + 1, i, str(v), va="center", fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("Tickets by Category", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Count"); axes[0].grid(axis="x", alpha=0.3)

    pc = df["priority"].value_counts().reindex(["High","Medium","Low"])
    axes[1].pie(pc.values, labels=pc.index, autopct="%1.1f%%",
                colors=[PRI_COLORS[p] for p in pc.index],
                startangle=140, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Priority Distribution (Engineered)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(candidate_out, "eda_overview.png"), dpi=150)
    plt.close(fig)
    print("  EDA chart saved")

    # ── Feature engineering (training-only fitting) ───────────────
    print("\n[2/6] Building features …")
    df = build_features(df)
    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        print("\nERROR: tickets_clean.csv must contain train and test split rows.")
        sys.exit(1)
    tfidf  = get_tfidf()
    X_tfidf = tfidf.fit_transform(train_df["clean_text"])
    scaler  = MaxAbsScaler()
    X_meta  = scaler.fit_transform(csr_matrix(train_df[META_COLS].values))
    X_tr    = hstack([X_tfidf, X_meta])

    X_test_tfidf = tfidf.transform(test_df["clean_text"])
    X_test_meta  = scaler.transform(csr_matrix(test_df[META_COLS].values))
    X_te         = hstack([X_test_tfidf, X_test_meta])
    X_cat_tr     = X_tfidf
    X_cat_te     = X_test_tfidf
    print(f"      Train feature matrix: {X_tr.shape}")
    print(f"      Test feature matrix:  {X_te.shape}")
    print(f"      Category text matrix: {X_cat_tr.shape}")

    # ── Encode labels ─────────────────────────────────────────────
    le_cat = LabelEncoder()
    le_pri = LabelEncoder()
    yc_tr  = le_cat.fit_transform(train_df["category"])
    yc_te  = le_cat.transform(test_df["category"])
    y_pri  = le_pri.fit_transform(train_df["priority"])
    yp_tr  = y_pri
    yp_te  = le_pri.transform(test_df["priority"])
    print(f"\n[3/6] Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}")

    min_class_size = train_df['category'].value_counts().min()
    n_splits = min(5, min_class_size)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    macro_f1 = make_scorer(f1_score, average="macro", zero_division=0)
    print(f"Using {n_splits}-fold cross-validation")
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, C=5, class_weight="balanced")),
        ("Linear SVC",          LinearSVC(max_iter=2000, C=1.0, class_weight="balanced")),
        ("Naive Bayes",         MultinomialNB(alpha=0.1)),
        ("Random Forest",       RandomForestClassifier(n_estimators=200, max_depth=20,
                                                        class_weight="balanced",
                                                        n_jobs=-1, random_state=42)),
    ]

    # ── Category ──────────────────────────────────────────────────
    print("\n[4/6] Category classifiers …")
    cat_res = []
    for name, clf in models:
        sc = cross_val_score(clf, X_cat_tr, yc_tr, cv=cv, scoring=macro_f1, n_jobs=-1)
        cat_res.append({"model": name, "mean": sc.mean(), "std": sc.std()})
        print(f"      {name:<25} {sc.mean():.4f} ± {sc.std():.4f}")

    plot_model_comparison(cat_res, "Category — Model Comparison (5-Fold CV)",
                          os.path.join(candidate_out, "cat_model_comparison.png"))

    best_cat_name = max(cat_res, key=lambda x: x["mean"])["model"]
    best_cat = clone(dict(models)[best_cat_name])
    best_cat.fit(X_cat_tr, yc_tr)
    yc_pred  = best_cat.predict(X_cat_te)
    cat_acc  = accuracy_score(yc_te, yc_pred)
    cat_rep  = classification_report(yc_te, yc_pred,
                                      target_names=le_cat.classes_, output_dict=True)
    print(f"\n  Best: {best_cat_name}  |  Test Acc: {cat_acc*100:.1f}%")
    print(classification_report(yc_te, yc_pred, target_names=le_cat.classes_))

    plot_confusion(yc_te, yc_pred, le_cat.classes_,
                   f"Category Confusion Matrix\nTest Accuracy: {cat_acc*100:.1f}%",
                   os.path.join(candidate_out, "cat_confusion_matrix.png"))
    plot_class_f1(cat_rep, list(le_cat.classes_),
                  "Category F1 per Class", COLORS,
                  os.path.join(candidate_out, "cat_f1_per_class.png"))

    # ── Priority ──────────────────────────────────────────────────
    print("\n[5/6] Priority classifiers …")
    pri_res = []
    for name, clf in models:
        sc = cross_val_score(clf, X_tr, yp_tr, cv=cv, scoring=macro_f1, n_jobs=-1)
        pri_res.append({"model": name, "mean": sc.mean(), "std": sc.std()})
        print(f"      {name:<25} {sc.mean():.4f} ± {sc.std():.4f}")

    plot_model_comparison(pri_res, "Priority — Model Comparison (5-Fold CV)",
                          os.path.join(candidate_out, "pri_model_comparison.png"))

    best_pri_name = max(pri_res, key=lambda x: x["mean"])["model"]
    best_pri = clone(dict(models)[best_pri_name])
    best_pri.fit(X_tr, yp_tr)
    yp_pred  = best_pri.predict(X_te)
    pri_acc  = accuracy_score(yp_te, yp_pred)
    pri_rep  = classification_report(yp_te, yp_pred,
                                      target_names=le_pri.classes_, output_dict=True)
    print(f"\n  Best: {best_pri_name}  |  Test Acc: {pri_acc*100:.1f}%")
    print(classification_report(yp_te, yp_pred, target_names=le_pri.classes_))

    plot_confusion(yp_te, yp_pred, le_pri.classes_,
                   f"Priority Confusion Matrix\nTest Accuracy: {pri_acc*100:.1f}%",
                   os.path.join(candidate_out, "pri_confusion_matrix.png"))
    plot_class_f1(pri_rep, list(le_pri.classes_),
                  "Priority F1 per Class",
                  [PRI_COLORS[c] for c in le_pri.classes_],
                  os.path.join(candidate_out, "pri_f1_per_class.png"))

    # ── Save and validate a candidate before activation ──────────
    print("\n[6/6] Saving and validating candidate models …")
    candidate_files = {
        "tfidf.pkl": tfidf,
        "scaler.pkl": scaler,
        "cat_model.pkl": best_cat,
        "pri_model.pkl": best_pri,
        "le_cat.pkl": le_cat,
        "le_pri.pkl": le_pri,
    }
    for filename, artifact in candidate_files.items():
        joblib.dump(artifact, os.path.join(candidate_dir, filename))

    loaded = {name: joblib.load(os.path.join(candidate_dir, filename))
              for name, filename in [
                  ("tfidf", "tfidf.pkl"), ("scaler", "scaler.pkl"),
                  ("cat", "cat_model.pkl"), ("pri", "pri_model.pkl"),
                  ("le_cat", "le_cat.pkl"), ("le_pri", "le_pri.pkl")
              ]}
    sample_text = "laptop got damage in the corner"
    sample_clean = clean_text(sample_text)
    sample_tfidf = loaded["tfidf"].transform([sample_clean])
    sample_meta = loaded["scaler"].transform(csr_matrix(pd.DataFrame([{
        "char_len": len(sample_text),
        "word_count": len(sample_text.split()),
        "exclaim": sample_text.count("!"),
        "question": sample_text.count("?"),
        "urgency_flag": int(bool(__import__("re").search(
            r"\b(?:urgent|asap|immediately|critical|emergency|broken|crash|"
            r"production|hack|compromised|down|outage)\b", sample_text.lower())))
    }])[META_COLS].values))
    sample_cat_X = sample_tfidf
    sample_pri_X = hstack([sample_tfidf, sample_meta])
    loaded["cat"].predict(sample_cat_X)
    loaded["pri"].predict(sample_pri_X)

    cat_macro_f1 = cat_rep["macro avg"]["f1-score"]
    cat_weighted_f1 = cat_rep["weighted avg"]["f1-score"]
    pri_macro_f1 = pri_rep["macro avg"]["f1-score"]
    pri_weighted_f1 = pri_rep["weighted avg"]["f1-score"]

    summary = {
        "cat_model": best_cat_name, "cat_accuracy": round(cat_acc, 4),
        "cat_macro_f1": round(cat_macro_f1, 4),
        "cat_weighted_f1": round(cat_weighted_f1, 4),
        "pri_model": best_pri_name, "pri_accuracy": round(pri_acc, 4),
        "pri_macro_f1": round(pri_macro_f1, 4),
        "pri_weighted_f1": round(pri_weighted_f1, 4),
        "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "cat_n_features": int(X_cat_tr.shape[1]),
        "categories": list(le_cat.classes_),
        "priorities":  list(le_pri.classes_),
        "cat_report": cat_rep, "pri_report": pri_rep,
        "priority_type": "rule-derived heuristic labels",
        "data_split": "supplied train/test CSVs",
        "model_selection_metric": "cross-validated macro-F1",
        "cat_cv": cat_res, "pri_cv": pri_res,
    }
    candidate_summary = os.path.join(candidate_out, "summary.json")
    with open(candidate_summary, "w") as f:
        json.dump(summary, f, indent=2)

    model_backup = os.path.join(MDL, "backups", timestamp)
    output_backup = os.path.join(OUT, "backups", timestamp)
    os.makedirs(model_backup, exist_ok=True)
    os.makedirs(output_backup, exist_ok=True)
    for filename in candidate_files:
        active = os.path.join(MDL, filename)
        if os.path.exists(active):
            shutil.copy2(active, os.path.join(model_backup, filename))
        shutil.copy2(os.path.join(candidate_dir, filename), active)

    active_summary = os.path.join(OUT, "summary.json")
    if os.path.exists(active_summary):
        shutil.copy2(active_summary, os.path.join(output_backup, "summary.json"))
    shutil.copy2(candidate_summary, active_summary)

    for filename in os.listdir(candidate_out):
        if filename.endswith(".png"):
            active_chart = os.path.join(OUT, filename)
            if os.path.exists(active_chart):
                shutil.copy2(active_chart, os.path.join(output_backup, filename))
            shutil.copy2(os.path.join(candidate_out, filename), active_chart)

    print("  Candidate validated and activated")
    print(f"  Previous models backed up -> {model_backup}")
    print(f"  Previous outputs backed up -> {output_backup}")

    print("\n" + "=" * 60)
    print(f"  Category Accuracy : {cat_acc*100:.1f}%  ({best_cat_name})")
    print(f"  Priority Accuracy : {pri_acc*100:.1f}%  ({best_pri_name})")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    main()
