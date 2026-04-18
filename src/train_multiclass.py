# src/train_multiclass.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from src.preprocess import run_preprocessing_pipeline
from src.vectorize import vectorize_data


# ============================================
# TRAIN RANDOM FOREST
# ============================================

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# ============================================
# TRAIN LINEAR SVM
# ============================================

def train_svm(X_train, y_train):
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model


# ============================================
# EVALUATE MODEL
# ============================================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


# ============================================
# SAVE MODEL
# ============================================

def save_model(model, filepath="models/multiclass_model.pkl"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Saved best model to {filepath}")


# ============================================
# COMPLETE PIPELINE
# ============================================

def run_multiclass_training_pipeline(save_model_flag=True):
    print("=" * 60)
    print("MULTICLASS MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Load + preprocess
    print("\nStep 1: Loading and preprocessing data...")
    df_binary, df_multi = run_preprocessing_pipeline(save_output=False)

    # Step 2: Vectorize
    print("\nStep 2: Vectorizing text...")
    X_train, X_test, y_train, y_test, vectorizer = vectorize_data(df_multi)

    # Step 3: Train models
    print("\nStep 3: Training models...")
    rf_model = train_random_forest(X_train, y_train)
    svm_model = train_svm(X_train, y_train)

    # Step 4: Evaluate
    print("\nStep 4: Evaluating models...")
    rf_acc = evaluate_model(rf_model, X_test, y_test)
    svm_acc = evaluate_model(svm_model, X_test, y_test)

    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"SVM Accuracy: {svm_acc:.4f}")

    # Step 5: Choose best model
    if rf_acc > svm_acc:
        best_model = rf_model
        best_name = "Random Forest"
        best_score = rf_acc
    else:
        best_model = svm_model
        best_name = "SVM"
        best_score = svm_acc

    print(f"\nBest Model: {best_name} ({best_score:.4f})")

    # Step 6: Save best model
    if save_model_flag:
        save_model(best_model)

    print("\n" + "=" * 60)
    print("MULTICLASS TRAINING COMPLETE")
    print("=" * 60)

    return best_model, X_test, y_test


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("Testing train_multiclass.py...")

    best_model, X_test, y_test = run_multiclass_training_pipeline(save_model_flag=False)

    print("\nMulticlass training test passed!")