# src/split_data.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import run_preprocessing_pipeline
from src.constants import (
    COL_TWEET,
    COL_CLEANED,
    COL_SENTIMENT,
    COL_TWEET_LENGTH,
    COL_WORD_COUNT,
    TEST_SIZE,
    RANDOM_STATE,
    FIGURE_SIZE,
    COLOR_BACKGROUND,
)

PLOTS_DIR = "figures"
os.makedirs(PLOTS_DIR, exist_ok=True)


### STRATIFIED TRAIN/TEST SPLIT


def stratified_train_test_split(df, label_col):
    """
    Split data into train and test sets using stratification
    to preserve class distribution.
    """
    X = df[COL_CLEANED]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train size : {len(X_train)}")
    print(f"Test size  : {len(X_test)}")
    print(f"Train class distribution:\n{y_train.value_counts()}")
    print(f"Test class distribution:\n{y_test.value_counts()}")

    return X_train, X_test, y_train, y_test


### TRAIN/VALIDATION/TEST SPLIT (60/20/20)

def train_val_test_split(df, label_col):
    """
    Split data into train (60%), validation (20%), and test (20%) sets
    with stratification to preserve class distribution.
    """
    X = df[COL_CLEANED]
    y = df[label_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    print(f"Train size     : {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Validation size: {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
    print(f"Test size      : {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


### DATA LEAKAGE CHECK

def check_data_leakage(X_train, X_val, X_test):
    """
    Check that no tweet appears in more than one split.
    Prints results and raises an error if leakage is found.
    """
    train_set = set(X_train)
    val_set   = set(X_val)
    test_set  = set(X_test)

    train_val_overlap  = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap   = val_set   & test_set

    print("Data Leakage Check")
    print("=" * 60)
    print(f"Train/Validation overlap : {len(train_val_overlap)}")
    print(f"Train/Test overlap       : {len(train_test_overlap)}")
    print(f"Validation/Test overlap  : {len(val_test_overlap)}")

    total_overlap = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)

    if total_overlap == 0:
        print("No data leakage found.")
    else:
        raise ValueError(f"Data leakage detected: {total_overlap} overlapping tweets found.")


### FEATURE CORRELATION HEATMAP

def plot_feature_correlation_heatmap(df):
    """
    Add tweet_length and word_count features then plot
    a correlation heatmap of numeric features.
    """
    df = df.copy()
    df[COL_TWEET_LENGTH] = df[COL_TWEET].str.len()
    df[COL_WORD_COUNT]   = df[COL_TWEET].str.split().str.len()

    df["sentiment_numeric"] = df[COL_SENTIMENT].map({ 
        "Positive emotion":2,
        "No emotion toward brand or product": 1,
        "Negative emotion": 0
    })
    numeric_cols = [COL_TWEET_LENGTH, COL_WORD_COUNT, "sentiment_numeric"]
    corr_matrix  = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=11)

    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_correlation_heatmap.png"), dpi=150)
    plt.close()



### COMPLETE PIPELINE

def run_splitting_pipeline():
    """Run all data splitting subtasks in order."""
    print("=" * 60)
    print("DATA SPLITTING PIPELINE")
    print("=" * 60)

    print("\nStep 1: Running preprocessing...")
    df_binary, df_multiclass = run_preprocessing_pipeline(save_output=False)
    df_binary = df_binary.drop_duplicates(subset=[COL_CLEANED]).reset_index(drop=True)
    df_multiclass = df_multiclass.drop_duplicates(subset=[COL_CLEANED]).reset_index(drop=True)

    print("\nStep 2: Stratified train/test split (binary)...")
    X_train, X_test, y_train, y_test = stratified_train_test_split(df_binary, "label_binary")

    print("\nStep 3: Train/validation/test split (binary 60/20/20)...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_binary, "label_binary")

    print("\nStep 4: Data leakage check...")
    check_data_leakage(X_train, X_val, X_test)

    print("\nStep 5: Feature correlation heatmap...")
    plot_feature_correlation_heatmap(df_binary)

    print("\n" + "=" * 60)
    print("DATA SPLITTING COMPLETE")
    print("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    run_splitting_pipeline()