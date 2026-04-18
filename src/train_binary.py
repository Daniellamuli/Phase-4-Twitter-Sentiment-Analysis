# src/train_binary.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from src.preprocess import run_preprocessing_pipeline
from src.vectorize import vectorize_data
from src.split_data import stratified_train_test_split
from src.constants import (
    BINARY_MODEL_PATH,
    LR_C,
    LR_MAX_ITER,
    LR_SOLVER,
    LR_PARAM_GRID,
    RANDOM_STATE
)


# ============================================
# TRAIN LOGISTIC REGRESSION
# ============================================

def train_logistic_regression(X_train, y_train, use_tuning=False):
    """Train Logistic Regression classifier for binary sentiment.
    
    Args:
        X_train: Training features (vectorized)
        y_train: Training labels (0=Negative, 1=Positive)
        use_tuning: If True, use GridSearchCV for hyperparameter tuning
    
    Returns:
        Trained Logistic Regression model
    """
    if use_tuning:
        print("Tuning Logistic Regression hyperparameters...")
        model = GridSearchCV(
            LogisticRegression(max_iter=LR_MAX_ITER, random_state=RANDOM_STATE),
            LR_PARAM_GRID,
            cv=5,
            scoring='f1'
        )
        model.fit(X_train, y_train)
        print(f"Best parameters: {model.best_params_}")
        print(f"Best CV score: {model.best_score_:.4f}")
        return model
    else:
        model = LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            solver=LR_SOLVER,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        return model


# ============================================
# TRAIN NAIVE BAYES
# ============================================

def train_naive_bayes(X_train, y_train):
    """Train Multinomial Naive Bayes classifier for binary sentiment.
    
    Args:
        X_train: Training features (vectorized)
        y_train: Training labels (0=Negative, 1=Positive)
    
    Returns:
        Trained Naive Bayes model
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


# ============================================
# SAVE MODEL
# ============================================

def save_model(model, filepath=BINARY_MODEL_PATH):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


# ============================================
# LOAD MODEL
# ============================================

def load_model(filepath=BINARY_MODEL_PATH):
    """Load saved model from disk."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


# ============================================
# COMPLETE TRAINING PIPELINE
# ============================================

def run_binary_training_pipeline(use_tuning=False, save_model_flag=True):
    """Run complete binary model training pipeline.
    
    Args:
        use_tuning: Whether to use hyperparameter tuning
        save_model_flag: Whether to save the trained model
    
    Returns:
        tuple: (logreg_model, nb_model, X_test, y_test, vectorizer)
    """
    print("=" * 60)
    print("BINARY MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    df_binary, df_multiclass = run_preprocessing_pipeline(save_output=False)
    
    # Step 2: Vectorize data
    print("\nStep 2: Vectorizing text data...")
    X_train, X_test, y_train, y_test, vectorizer = vectorize_data(df_binary)
    
    # Step 3: Train Logistic Regression
    print("\nStep 3: Training Logistic Regression...")
    logreg_model = train_logistic_regression(X_train, y_train, use_tuning=use_tuning)
    
    # Step 4: Train Naive Bayes
    print("\nStep 4: Training Naive Bayes...")
    nb_model = train_naive_bayes(X_train, y_train)
    
    # Step 5: Save models
    if save_model_flag:
        print("\nStep 5: Saving models...")
        save_model(logreg_model, "models/logreg_binary.pkl")
        save_model(nb_model, "models/nb_binary.pkl")
    
    print("\n" + "=" * 60)
    print("BINARY TRAINING COMPLETE")
    print("=" * 60)
    
    return logreg_model, nb_model, X_test, y_test, vectorizer


# ============================================
# TEST CODE
# ============================================

if __name__ == "__main__":
    print("Testing train_binary.py...")
    
    # Run training without tuning for quick test
    logreg_model, nb_model, X_test, y_test, vectorizer = run_binary_training_pipeline(
        use_tuning=False, 
        save_model_flag=False
    )
    
    # Quick evaluation
    from sklearn.metrics import accuracy_score
    
    y_pred_logreg = logreg_model.predict(X_test)
    y_pred_nb = nb_model.predict(X_test)
    
    print("\n" + "-" * 40)
    print("QUICK TEST RESULTS")
    print("-" * 40)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    
    print("\nTraining test passed!")