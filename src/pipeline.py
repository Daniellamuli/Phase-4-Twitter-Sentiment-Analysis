# src/pipeline.py
"""
Formalised end-to-end sklearn Pipeline objects.
Each pipeline bundles TF-IDF vectorisation + optional class-weight
adjustment + estimator into a single, cross-validation-friendly object.

Usage:
    from src.pipeline import (
        build_lr_pipeline,
        build_nb_pipeline,
        build_rf_pipeline,
        build_svm_pipeline,
        run_pipeline_comparison,
    )
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, f1_score,
    accuracy_score, precision_score, recall_score,
)

from src.constants import (
    MAX_FEATURES, NGRAM_RANGE, TEST_SIZE, RANDOM_STATE,
    LR_C, LR_MAX_ITER, LR_SOLVER,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
    COL_CLEANED,
)


# ============================================================
# PIPELINE BUILDERS
# ============================================================

def build_lr_pipeline(class_weight="balanced"):
    """
    Logistic Regression pipeline with TF-IDF.

    class_weight="balanced" automatically up-weights the Negative class
    inversely proportional to its frequency — the simplest and most
    effective fix for class imbalance with linear models.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
        )),
        ("clf", LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            solver=LR_SOLVER,
            class_weight=class_weight,   # <-- key imbalance fix
            random_state=RANDOM_STATE,
        )),
    ])


def build_nb_pipeline():
    """
    Multinomial Naive Bayes pipeline with TF-IDF.

    MultinomialNB does not support class_weight directly.
    Class imbalance is partially mitigated by its generative
    probability estimates, which are less sensitive to skewed
    class frequencies than discriminative models.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
        )),
        ("clf", MultinomialNB()),
    ])


def build_rf_pipeline(class_weight="balanced"):
    """
    Random Forest pipeline with TF-IDF.

    class_weight="balanced" adjusts sample weights at each tree split,
    giving more influence to the minority Negative class.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            class_weight=class_weight,   # <-- key imbalance fix
            random_state=RF_RANDOM_STATE,
        )),
    ])


def build_svm_pipeline(class_weight="balanced"):
    """
    Linear SVM pipeline with TF-IDF.

    class_weight="balanced" is particularly effective for LinearSVC on
    imbalanced text data — it penalises misclassifying the minority class
    more heavily, directly improving Negative class recall.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
        )),
        ("clf", LinearSVC(
            class_weight=class_weight,   # <-- key imbalance fix
            random_state=RANDOM_STATE,
            max_iter=2000,
        )),
    ])


# ============================================================
# EVALUATION HELPER
# ============================================================

def evaluate_pipeline(name, pipeline, X_test, y_test):
    """Run predictions from a fitted pipeline and return a metrics dict."""
    y_pred = pipeline.predict(X_test)
    return {
        "name":      name,
        "pipeline":  pipeline,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred,    average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred,        average="weighted", zero_division=0),
        "report":    classification_report(y_test, y_pred, zero_division=0),
    }


# ============================================================
# CROSS-VALIDATION HELPER
# ============================================================

def cross_validate_pipeline(name, pipeline, X, y, cv=5):
    """
    Run stratified k-fold cross-validation on raw text + labels.
    The pipeline handles vectorisation internally on each fold,
    preventing data leakage between train and validation splits.
    """
    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    print(f"{name}")
    print(f"  CV F1 (weighted) — mean: {scores.mean():.4f}  std: {scores.std():.4f}")
    print(f"  Per-fold scores  : {[round(s, 4) for s in scores]}")
    return scores


# ============================================================
# COMPLETE PIPELINE COMPARISON
# ============================================================

def run_pipeline_comparison(df_binary, df_multiclass):
    """
    Train, cross-validate, and evaluate all four pipelines.
    Prints a ranked comparison table and returns the results list.
    """
    print("=" * 60)
    print("PIPELINE COMPARISON (with class_weight='balanced')")
    print("=" * 60)

    # ── Binary splits ─────────────────────────────────────────
    X_b = df_binary[COL_CLEANED]
    y_b = df_binary["label_binary"]
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_b, y_b, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_b
    )

    # ── Multiclass splits ─────────────────────────────────────
    X_m = df_multiclass[COL_CLEANED]
    y_m = df_multiclass["label_multiclass"]
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_m, y_m, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_m
    )

    # ── Build pipelines ───────────────────────────────────────
    pipelines = {
        "Logistic Regression (Binary, balanced)":  (build_lr_pipeline(),  X_train_b, y_train_b, X_test_b, y_test_b),
        "Naive Bayes (Binary)":                    (build_nb_pipeline(),  X_train_b, y_train_b, X_test_b, y_test_b),
        "Random Forest (Multiclass, balanced)":    (build_rf_pipeline(),  X_train_m, y_train_m, X_test_m, y_test_m),
        "SVM (Multiclass, balanced)":              (build_svm_pipeline(), X_train_m, y_train_m, X_test_m, y_test_m),
    }

    # ── Cross-validation ──────────────────────────────────────
    print("\n--- 5-Fold Cross-Validation (F1 weighted) ---")
    cross_validate_pipeline("Logistic Regression (Binary)",  build_lr_pipeline(),  X_b, y_b)
    cross_validate_pipeline("Naive Bayes (Binary)",           build_nb_pipeline(),  X_b, y_b)
    cross_validate_pipeline("Random Forest (Multiclass)",     build_rf_pipeline(),  X_m, y_m)
    cross_validate_pipeline("SVM (Multiclass)",               build_svm_pipeline(), X_m, y_m)

    # ── Train & evaluate ──────────────────────────────────────
    print("\n--- Test Set Evaluation ---")
    results = []
    for name, (pipe, X_tr, y_tr, X_te, y_te) in pipelines.items():
        pipe.fit(X_tr, y_tr)
        r = evaluate_pipeline(name, pipe, X_te, y_te)
        results.append(r)
        print(f"\n{name}")
        print(r["report"])

    # ── Ranked summary table ──────────────────────────────────
    ranked = sorted(results, key=lambda r: r["f1"], reverse=True)
    print("\n--- Model Ranking by F1 Score ---")
    print("=" * 60)
    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r['name']:<45} F1: {r['f1']:.4f}")

    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    from src.preprocess import run_preprocessing_pipeline
    df_binary, df_multiclass = run_preprocessing_pipeline(save_output=False)
    run_pipeline_comparison(df_binary, df_multiclass)



