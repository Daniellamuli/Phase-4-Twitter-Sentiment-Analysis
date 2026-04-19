# src/export_tableau_data.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.preprocess import run_preprocessing_pipeline
from src.vectorize import vectorize_data
from src.train_binary import train_logistic_regression, train_naive_bayes
from src.train_multiclass import train_random_forest, train_svm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

TABLEAU_DIR = "tableau"
os.makedirs(TABLEAU_DIR, exist_ok=True)



# 1. SENTIMENT SUMMARY


def export_sentiment_summary(df):
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    counts["percentage"] = (counts["count"] / counts["count"].sum() * 100).round(2)

    # Clean up label names for Tableau
    counts["sentiment_label"] = counts["sentiment"].map({
        "Positive emotion":                    "Positive",
        "Negative emotion":                    "Negative",
        "No emotion toward brand or product":  "Neutral",
    })

    path = os.path.join(TABLEAU_DIR, "sentiment_summary.csv")
    counts.to_csv(path, index=False)
    print(f"Saved: {path}")



# 2. TOP PRODUCTS


def export_top_products(df):
    top = df["product"].value_counts().head(10).reset_index()
    top.columns = ["product", "mention_count"]

    path = os.path.join(TABLEAU_DIR, "top_products.csv")
    top.to_csv(path, index=False)
    print(f"Saved: {path}")



# 3. APPLE VS GOOGLE SENTIMENT


def export_apple_vs_google(df):
    apple_keywords  = ["iphone", "ipad", "ipod", "macbook", "apple", "imac"]
    google_keywords = ["android", "google", "gmail", "chrome"]

    def get_brand(product):
        if pd.isna(product):
            return None
        product_lower = product.lower()
        if any(k in product_lower for k in apple_keywords):
            return "Apple"
        if any(k in product_lower for k in google_keywords):
            return "Google"
        return None

    df = df.copy()
    df["brand"] = df["product"].apply(get_brand)
    df = df[df["brand"].notna()]

    df["sentiment_label"] = df["sentiment"].map({
        "Positive emotion":                   "Positive",
        "Negative emotion":                   "Negative",
        "No emotion toward brand or product": "Neutral",
    })

    summary = df.groupby(["brand", "sentiment_label"]).size().reset_index(name="count")
    summary["percentage"] = summary.groupby("brand")["count"].transform(
        lambda x: (x / x.sum() * 100).round(2)
    )

    path = os.path.join(TABLEAU_DIR, "apple_vs_google.csv")
    summary.to_csv(path, index=False)
    print(f"Saved: {path}")


# 4. MODEL RESULTS


def export_model_results(df_binary, df_multiclass):
    def evaluate(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            "model":      name,
            "accuracy":   round(accuracy_score(y_test, y_pred), 4),
            "precision":  round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "recall":     round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "f1_score":   round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "type":       "Binary" if "Binary" in name else "Multiclass",
        }

    X_train_b, X_test_b, y_train_b, y_test_b, _ = vectorize_data(df_binary)
    X_train_m, X_test_m, y_train_m, y_test_m, _ = vectorize_data(df_multiclass)

    results = [
        evaluate("Logistic Regression (Binary)",  train_logistic_regression(X_train_b, y_train_b), X_test_b, y_test_b),
        evaluate("Naive Bayes (Binary)",           train_naive_bayes(X_train_b, y_train_b),         X_test_b, y_test_b),
        evaluate("Random Forest (Multiclass)",     train_random_forest(X_train_m, y_train_m),       X_test_m, y_test_m),
        evaluate("SVM (Multiclass)",               train_svm(X_train_m, y_train_m),                 X_test_m, y_test_m),
    ]

    df_results = pd.DataFrame(results)
    path = os.path.join(TABLEAU_DIR, "model_results.csv")
    df_results.to_csv(path, index=False)
    print(f"Saved: {path}")


# 5. FULL TWEET DATA (for word-level analysis)


def export_tweet_data(df):
    df = df.copy()
    df["sentiment_label"] = df["sentiment"].map({
        "Positive emotion":                   "Positive",
        "Negative emotion":                   "Negative",
        "No emotion toward brand or product": "Neutral",
    })

    export_cols = ["tweet", "product", "sentiment_label", "cleaned_text"]
    export_cols = [c for c in export_cols if c in df.columns]

    path = os.path.join(TABLEAU_DIR, "tweets_clean.csv")
    df[export_cols].to_csv(path, index=False)
    print(f"Saved: {path}")



# COMPLETE EXPORT PIPELINE


def run_export_pipeline():
    print("=" * 60)
    print("TABLEAU DATA EXPORT PIPELINE")
    print("=" * 60)

    print("\nStep 1: Loading and preprocessing data...")
    df_binary, df_multiclass = run_preprocessing_pipeline(save_output=False)

    # Get full df for general exports
    from src.load_data import load_and_prepare_data
    from src.clean_text import clean_dataframe
    df = load_and_prepare_data()
    df = clean_dataframe(df)

    print("\nStep 2: Exporting sentiment summary...")
    export_sentiment_summary(df)

    print("\nStep 3: Exporting top products...")
    export_top_products(df)

    print("\nStep 4: Exporting Apple vs Google breakdown...")
    export_apple_vs_google(df)

    print("\nStep 5: Exporting model results...")
    export_model_results(df_binary, df_multiclass)

    print("\nStep 6: Exporting clean tweet data...")
    export_tweet_data(df)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE — files saved to tableau/")
    print("=" * 60)


if __name__ == "__main__":
    run_export_pipeline()