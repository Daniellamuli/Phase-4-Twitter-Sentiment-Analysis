# src/pipeline.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports
from src.preprocess import run_preprocessing_pipeline
from src.features import add_all_features
from src.train_multiclass import run_multiclass_training_pipeline
from src.evaluate import evaluate_model, plot_confusion_matrix


# ============================================
# FULL PIPELINE
# ============================================

def run_full_pipeline():
    print("=" * 60)
    print("FULL ML PIPELINE")
    print("=" * 60)

    # ----------------------------------------
    # STEP 1: PREPROCESS DATA
    # ----------------------------------------
    print("\nStep 1: Preprocessing data...")
    df_binary, df_multi = run_preprocessing_pipeline(save_output=False)

    # ----------------------------------------
    # STEP 2: FEATURE ENGINEERING
    # ----------------------------------------
    print("\nStep 2: Adding features...")
    df_multi = add_all_features(df_multi)

    # ----------------------------------------
    # STEP 3: TRAIN MODEL (includes vectorization)
    # ----------------------------------------
    print("\nStep 3: Training multiclass model...")
    best_model, X_test, y_test = run_multiclass_training_pipeline(save_model_flag=False)

    # ----------------------------------------
    # STEP 4: PREDICT
    # ----------------------------------------
    print("\nStep 4: Generating predictions...")
    y_pred = best_model.predict(X_test)

    # ----------------------------------------
    # STEP 5: EVALUATE
    # ----------------------------------------
    print("\nStep 5: Evaluating model...")
    evaluate_model(y_test, y_pred)

    plot_confusion_matrix(
        y_test,
        y_pred,
        labels=["Negative", "Neutral", "Positive"]
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return best_model


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    run_full_pipeline()