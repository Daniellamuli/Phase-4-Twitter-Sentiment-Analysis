# src/compare_models.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from src.preprocess import run_preprocessing_pipeline
from src.vectorize import vectorize_data
from src.train_binary import train_logistic_regression, train_naive_bayes
from src.train_multiclass import train_random_forest, train_svm
from src.constants import (
    FIGURE_SIZE,
    COLOR_BACKGROUND,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
)

PLOTS_DIR = "figures"
os.makedirs(PLOTS_DIR, exist_ok=True)



# MODEL RANKING & BEST MODEL SELECTION


def rank_models(results):
    """
    Rank models by F1 score and return the best model name.
    results: list of dicts with keys: name, model, f1, accuracy, precision, recall
    """
    ranked = sorted(results, key=lambda x: x["f1"], reverse=True)
    best = ranked[0]

    print("Model Ranking by F1 Score")
    print("=" * 60)
    for i, r in enumerate(ranked, start=1):
        print(f"{i}. {r['name']:<30} F1: {r['f1']:.4f}")

    print(f"\nBest Model: {best['name']} (F1: {best['f1']:.4f})")

    return best, ranked



# MODEL COMPARISON TABLE


def build_comparison_table(results):
    """
    Build and print a comparison table of all models and their metrics.
    """
    rows = []
    for r in results:
        rows.append({
            "Model":     r["name"],
            "Accuracy":  round(r["accuracy"],  4),
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1 Score":  round(r["f1"],        4),
        })

    df_table = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)

    print("\nModel Comparison Table")
    print("=" * 60)
    print(df_table.to_string(index=False))

    return df_table



# BAR CHART COMPARING MODEL PERFORMANCE


def plot_model_comparison(results):
    """
    Plot a grouped bar chart comparing accuracy, precision, recall,
    and F1 score for all models side by side.
    """
    model_names = [r["name"] for r in results]
    metrics     = ["accuracy", "precision", "recall", "f1"]
    colors      = [COLOR_POSITIVE, COLOR_NEUTRAL, COLOR_NEGATIVE, "#3498db"]
    bar_width   = 0.18
    x           = range(len(model_names))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values  = [r[metric] for r in results]
        offsets = [pos + i * bar_width for pos in x]
        ax.bar(offsets, values, width=bar_width, label=metric.capitalize(), color=color, edgecolor="white")

    ax.set_xticks([pos + bar_width * 1.5 for pos in x])
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()


# RESULTS SUMMARY


def print_results_summary(best, ranked, df_table):
    """
    Print a written summary of the best model, its metrics, and limitations.
    """
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nBest Model  : {best['name']}")
    print(f"Accuracy    : {best['accuracy']:.4f}")
    print(f"Precision   : {best['precision']:.4f}")
    print(f"Recall      : {best['recall']:.4f}")
    print(f"F1 Score    : {best['f1']:.4f}")

    print("\nKey Findings:")
    print(f"  - {ranked[0]['name']} performed best overall based on F1 score.")
    print(f"  - {ranked[-1]['name']} had the lowest F1 score of {ranked[-1]['f1']:.4f}.")
    print(f"  - All models were trained on TF-IDF vectorized tweet text.")

    print("\nLimitations:")
    print("  - Dataset is imbalanced: negative tweets are underrepresented.")
    print("  - Models trained only on SXSW tweets may not generalize well.")
    print("  - TF-IDF does not capture word order or context.")

    print("\n" + "=" * 60)



# HELPER: EVALUATE A MODEL


def evaluate(name, model, X_test, y_test):
    """Run predictions and return a results dict."""
    y_pred = model.predict(X_test)
    return {
        "name":      name,
        "model":     model,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }



# COMPLETE PIPELINE


def run_comparison_pipeline():
    """Run all model comparison subtasks in order."""
    print("=" * 60)
    print("MODEL COMPARISON PIPELINE")
    print("=" * 60)

    ### Load and preprocess
    print("\nStep 1: Loading and preprocessing data...")
    df_binary, df_multiclass = run_preprocessing_pipeline(save_output=False)

    ###Vectorize for binary models
    print("\nStep 2: Vectorizing for binary models...")
    X_train_b, X_test_b, y_train_b, y_test_b, _ = vectorize_data(df_binary)

    #### Vectorize for multiclass models
    print("\nStep 3: Vectorizing for multiclass models...")
    X_train_m, X_test_m, y_train_m, y_test_m, _ = vectorize_data(df_multiclass)

    ### Train all models
    print("\nStep 4: Training all models...")
    lr_model  = train_logistic_regression(X_train_b, y_train_b)
    nb_model  = train_naive_bayes(X_train_b, y_train_b)
    rf_model  = train_random_forest(X_train_m, y_train_m)
    svm_model = train_svm(X_train_m, y_train_m)

    ### Evaluate all models
    print("\nStep 5: Evaluating all models...")
    results = [
        evaluate("Logistic Regression (Binary)",  lr_model,  X_test_b, y_test_b),
        evaluate("Naive Bayes (Binary)",           nb_model,  X_test_b, y_test_b),
        evaluate("Random Forest (Multiclass)",     rf_model,  X_test_m, y_test_m),
        evaluate("SVM (Multiclass)",               svm_model, X_test_m, y_test_m),
    ]

    ### Rank models
    print("\nStep 6: Ranking models...")
    best, ranked = rank_models(results)

    ### Comparison table
    print("\nStep 7: Building comparison table...")
    df_table = build_comparison_table(results)

    ###  Bar chart
    print("\nStep 8: Plotting model comparison chart...")
    plot_model_comparison(results)

    ### Results summary
    print("\nStep 9: Writing results summary...")
    print_results_summary(best, ranked, df_table)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 60)

    return best, df_table


if __name__ == "__main__":
    run_comparison_pipeline()