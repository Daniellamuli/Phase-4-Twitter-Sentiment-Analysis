#src/visualize.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from wordcloud import WordCloud
from src.constants import (    
    COL_TWEET,
    COL_PRODUCT,
    COL_SENTIMENT,
    COL_CLEANED,
    POSITIVE,
    NEGATIVE,
    NEUTRAL,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_BACKGROUND,
    FIGURE_SIZE,
    WORDCLOUD_WIDTH,
    WORDCLOUD_HEIGHT,
    WORDCLOUD_MAX_WORDS,
    WORDCLOUD_BACKGROUND,
    )

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok = True)

##DATA DICTIONARY


def document_data_dictionary(df):
    col_descriptions = {
        COL_TWEET:     "Raw tweet text from SXSW conference",
        COL_PRODUCT:   "Brand or product the emotion is directed at",
        COL_SENTIMENT: "Human-annotated sentiment label",
        COL_CLEANED:   "Cleaned and preprocessed tweet text",
    }
 
    print("DATA DICTIONARY")
    print("=" * 60)
 
    for col in df.columns:
        print(f"\nColumn     : {col}")
        print(f"Description: {col_descriptions.get(col, 'N/A')}")
        print(f"Data type  : {df[col].dtype}")
        print(f"Missing    : {df[col].isnull().sum()} ({df[col].isnull().mean() * 100:.1f}%)")
        print(f"Unique vals: {df[col].nunique()}")
        print(f"Sample vals: {df[col].dropna().unique()[:3].tolist()}")
