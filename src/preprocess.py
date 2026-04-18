# src/preprocess.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.load_data import load_and_prepare_data
from src.clean_text import clean_text  # Import just the clean_text function
from src.constants import (
    COL_SENTIMENT,
    COL_TWEET,
    COL_CLEANED,
    BINARY_LABEL_MAP,
    MULTICLASS_LABEL_MAP,
    PROCESSED_DATA_PATH
)


# Temporary cleaning function 
def clean_dataframe(df):
    """Apply cleaning to dataframe."""
    df[COL_CLEANED] = df[COL_TWEET].apply(clean_text)
    print(f"Cleaned {len(df)} tweets")
    return df


# ============================================
# ADD BINARY LABELS
# ============================================

def add_binary_labels(df):
    """Add numeric labels for binary classification."""
    df['label_binary'] = df[COL_SENTIMENT].map(BINARY_LABEL_MAP)
    df_binary = df.dropna(subset=['label_binary']).copy()
    df_binary['label_binary'] = df_binary['label_binary'].astype(int)
    
    print(f"Binary dataset: {len(df_binary)} rows")
    print(f"  - Positive (1): {(df_binary['label_binary'] == 1).sum()}")
    print(f"  - Negative (0): {(df_binary['label_binary'] == 0).sum()}")
    
    return df_binary


# ============================================
# ADD MULTICLASS LABELS
# ============================================

def add_multiclass_labels(df):
    """Add numeric labels for multiclass classification."""
    df['label_multiclass'] = df[COL_SENTIMENT].map(MULTICLASS_LABEL_MAP)
    df_multiclass = df[df['label_multiclass'] != -1].copy()
    df_multiclass['label_multiclass'] = df_multiclass['label_multiclass'].astype(int)
    
    print(f"Multiclass dataset: {len(df_multiclass)} rows")
    print(f"  - Positive (2): {(df_multiclass['label_multiclass'] == 2).sum()}")
    print(f"  - Neutral (1): {(df_multiclass['label_multiclass'] == 1).sum()}")
    print(f"  - Negative (0): {(df_multiclass['label_multiclass'] == 0).sum()}")
    
    return df_multiclass


# ============================================
# SAVE PROCESSED DATA
# ============================================

def save_processed_data(df, filepath=PROCESSED_DATA_PATH):
    """Save processed dataframe to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved processed data to {filepath}")
    return df


# ============================================
# MAIN PREPROCESSING PIPELINE
# ============================================

def run_preprocessing_pipeline(save_output=True):
    """Run the complete preprocessing pipeline."""
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    print("\nStep 1: Loading and cleaning data...")
    df = load_and_prepare_data()
    df = clean_dataframe(df)
    
    print("\nStep 2: Adding binary labels...")
    df_binary = add_binary_labels(df)
    
    print("\nStep 3: Adding multiclass labels...")
    df_multiclass = add_multiclass_labels(df)
    
    if save_output:
        print("\nStep 4: Saving processed data...")
        save_processed_data(df)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return df_binary, df_multiclass


if __name__ == "__main__":
    print("Testing preprocess.py...")
    df_binary, df_multiclass = run_preprocessing_pipeline(save_output=False)
    print("\nPreprocessing test passed!")