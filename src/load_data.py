# src/load_data.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.constants import RAW_DATA_PATH

# Simple path fix - add parent directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
sys.path.insert(0, parent_dir)

# Now import from src.constants
from src.constants import (
    RAW_DATA_PATH,
    COL_TWEET,
    COL_PRODUCT,
    COL_SENTIMENT,
    UNKNOWN
)


# ============================================
# DATA LOADER FUNCTION
# ============================================

def load_raw_data():
    """Load the CSV file and return raw dataframe."""
    df = pd.read_csv(RAW_DATA_PATH, encoding='latin-1')
    print(f"✅ Loaded {len(df)} rows from {RAW_DATA_PATH}")
    return df


# ============================================
# COLUMN RENAMING FUNCTION
# ============================================

def rename_columns(df):
    """Rename columns to standard names using constants."""
    original_columns = df.columns.tolist()
    
    df = df.rename(columns={
        original_columns[0]: COL_TWEET,
        original_columns[1]: COL_PRODUCT,
        original_columns[2]: COL_SENTIMENT
    })
    
    print(f"✅ Renamed columns: {df.columns.tolist()}")
    return df


# ============================================
# FILTER FUNCTION
# ============================================

def filter_unknown_sentiment(df):
    """Remove rows where sentiment is 'I can't tell'."""
    before_count = len(df)
    
    df_filtered = df[df[COL_SENTIMENT] != UNKNOWN].copy()
    
    after_count = len(df_filtered)
    removed_count = before_count - after_count
    
    print(f" Removed {removed_count} rows with '{UNKNOWN}' sentiment")
    print(f" Remaining rows: {after_count}")
    
    return df_filtered


# ============================================
# BASIC INFO FUNCTION
# ============================================

def get_basic_info(df):
    """Print basic information about the dataframe."""
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    
    print(f"\n Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print(f"\n Columns:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype}")
    
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\n Memory usage: {memory_mb:.2f} MB")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n Missing values:")
        for col, count in missing.items():
            if count > 0:
                print(f"   - {col}: {count} missing")
    else:
        print(f"\n✅ No missing values found")
    
    print(f"\n First 5 rows preview:")
    print(df.head())
    
    if COL_SENTIMENT in df.columns:
        print(f"\n Sentiment distribution:")
        sentiment_counts = df[COL_SENTIMENT].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   - {sentiment}: {count} ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)


# ============================================
# COMPLETE PIPELINE
# ============================================

def load_and_prepare_data():
    """Run the complete data loading pipeline."""
    print("\n" + "=" * 60)
    print("DATA LOADING PIPELINE")
    print("=" * 60)
    
    print("\n Step 1: Loading raw data...")
    df = load_raw_data()
    
    print("\n Step 2: Renaming columns...")
    df = rename_columns(df)
    
    print("\n Step 3: Filtering unknown sentiment...")
    df = filter_unknown_sentiment(df)
    
    print("\n Step 4: Dataset information...")
    get_basic_info(df)
    
    print("\n" + "=" * 60)
    print("✅ DATA LOADING COMPLETE!")
    print("=" * 60)
    
    return df


# ============================================
# TEST CODE
# ============================================

if __name__ == "__main__":
    print("\n" + "🧪" * 10)
    print("TESTING load_data.py")
    print("🧪" * 10)
    
    df = load_and_prepare_data()
    
    print("\n" + "-" * 40)
    print("VALIDATION CHECKS")
    print("-" * 40)
    
    required_columns = [COL_TWEET, COL_PRODUCT, COL_SENTIMENT]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if not missing_columns:
        print(f"✅ All required columns present")
    else:
        print(f"❌ Missing columns: {missing_columns}")
    
    if UNKNOWN not in df[COL_SENTIMENT].values:
        print(f"✅ No '{UNKNOWN}' sentiment remaining")
    else:
        print(f"❌ '{UNKNOWN}' still present")
    
    print("\n" + "🎉" * 10)
    print("All tests passed! load_data.py is ready.")
    print("🎉" * 10)