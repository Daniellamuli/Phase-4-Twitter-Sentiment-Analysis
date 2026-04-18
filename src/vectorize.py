# src/vectorize.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os

from src.constants import COL_CLEANED


# ============================================
# TF-IDF VECTORIZER
# ============================================

def build_tfidf_vectorizer(max_features=3000, ngram_range=(1, 2)):
    """Create TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )


# ============================================
# COUNT VECTORIZER
# ============================================

def build_count_vectorizer(max_features=3000, ngram_range=(1, 2)):
    """Create CountVectorizer."""
    return CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range
    )


# ============================================
# TRAIN / TEST SPLIT + VECTORIZATION
# ============================================

def vectorize_data(df, vectorizer_type="tfidf", test_size=0.2):
    """
    Split data and apply vectorization.
    """

    X = df[COL_CLEANED]
    y = df['label_multiclass'] if 'label_multiclass' in df.columns else df['label_binary']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Choose vectorizer
    if vectorizer_type == "tfidf":
        vectorizer = build_tfidf_vectorizer()
    else:
        vectorizer = build_count_vectorizer()

    # Fit on training data
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform test data
    X_test_vec = vectorizer.transform(X_test)

    print(f"Vectorization complete using {vectorizer_type.upper()}")
    print(f"Train shape: {X_train_vec.shape}")
    print(f"Test shape: {X_test_vec.shape}")

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


# ============================================
# SAVE VECTORIZER
# ============================================

def save_vectorizer(vectorizer, filepath="models/vectorizer.pkl"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(vectorizer, filepath)
    print(f"Vectorizer saved to {filepath}")


# ============================================
# LOAD VECTORIZER
# ============================================

def load_vectorizer(filepath="models/vectorizer.pkl"):
    vectorizer = joblib.load(filepath)
    print(f"Vectorizer loaded from {filepath}")
    return vectorizer



if __name__ == "__main__":
    from src.preprocess import run_preprocessing_pipeline

    df_binary, df_multi = run_preprocessing_pipeline(save_output=False)

    X_train, X_test, y_train, y_test, vectorizer = vectorize_data(df_multi)

    print("Vectorization test complete!")