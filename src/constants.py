"""Shared constants for Twitter Sentiment Analysis project.
All team members import from this file to ensure consistency across modules.
"""
#=========================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================
# FILE PATHS
# ============================================

# Raw data file
RAW_DATA_PATH = "data/judge-1377884607_tweet_product_company.csv"

# Processed data output
PROCESSED_DATA_PATH = "data/processed/cleaned_tweets.csv"

# Saved models
BINARY_MODEL_PATH = "models/binary_model.pkl"
MULTICLASS_MODEL_PATH = "models/multiclass_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


# ============================================
# RAW COLUMN NAMES (from the CSV file)
# ============================================

RAW_COL_TWEET = "tweet_text"
RAW_COL_PRODUCT = "emotion_in_tweet_is_directed_at"
RAW_COL_SENTIMENT = "is_there_an_emotion_directed_at_a_brand_or_product"


# ============================================
# RENAMED COLUMN NAMES (after cleaning)
# ============================================

COL_TWEET = "tweet"
COL_PRODUCT = "product"
COL_SENTIMENT = "sentiment"
COL_CLEANED = "cleaned_text"
COL_TWEET_LENGTH = "tweet_length"
COL_WORD_COUNT = "word_count"


# ============================================
# SENTIMENT LABELS (as they appear in the CSV)
# ============================================

POSITIVE = "Positive emotion"
NEGATIVE = "Negative emotion"
NEUTRAL = "No emotion toward brand or product"
UNKNOWN = "I can't tell"


# ============================================
# NUMERIC LABELS FOR MODELING
# ============================================

# Binary classification (Positive vs Negative only)
BINARY_LABEL_MAP = {
    POSITIVE: 1,   # Positive sentiment
    NEGATIVE: 0    # Negative sentiment
}

# Multiclass classification (Positive, Neutral, Negative)
MULTICLASS_LABEL_MAP = {
    POSITIVE: 2,   # Positive sentiment
    NEUTRAL: 1,    # Neutral sentiment
    NEGATIVE: 0,   # Negative sentiment
    UNKNOWN: -1    # Exclude from training
}


# ============================================
# MODEL PARAMETERS
# ============================================

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Vectorizer settings
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)  # Unigrams and bigrams

# Logistic Regression
LR_C = 1.0
LR_MAX_ITER = 1000
LR_SOLVER = "lbfgs"

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42

# GridSearchCV tuning parameters
LR_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0],
    "solver": ["lbfgs", "liblinear"]
}

RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10]
}


# ============================================
# TEXT CLEANING SETTINGS
# ============================================

REMOVE_URLS = True
REMOVE_MENTIONS = True
REMOVE_HASHTAG_SYMBOLS = True  # Remove # but keep the word
REMOVE_NUMBERS = True
REMOVE_PUNCTUATION = True
MIN_WORD_LENGTH = 2
USE_LEMMATIZATION = True


# ============================================
# VISUALIZATION SETTINGS
# ============================================

FIGURE_SIZE = (12, 8)
COLOR_POSITIVE = "#2ecc71"   # Green
COLOR_NEGATIVE = "#e74c3c"   # Red
COLOR_NEUTRAL = "#f39c12"    # Orange
COLOR_BACKGROUND = "#f8f9fa"

# Word cloud settings
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400
WORDCLOUD_MAX_WORDS = 100
WORDCLOUD_BACKGROUND = "white"


# ============================================
# EVALUATION METRICS
# ============================================

# Primary metrics for model selection
PRIMARY_METRIC_BINARY = "f1"
PRIMARY_METRIC_MULTICLASS = "weighted_f1"

# All metrics to track
METRICS = ["accuracy", "precision", "recall", "f1"]


# ============================================
# PRODUCT LIST (for analysis)
# ============================================

APPLE_PRODUCTS = ["iphone", "ipad", "ipod", "macbook", "apple watch", "imac"]
GOOGLE_PRODUCTS = ["android", "google maps", "gmail", "google docs", "chrome"]


# ============================================
# POSITIVE AND NEGATIVE WORD LISTS
# ============================================

POSITIVE_WORDS = {
    "love", "great", "awesome", "amazing", "good", "best", "excellent",
    "fantastic", "perfect", "wonderful", "happy", "beautiful", "cool",
    "nice", "brilliant", "genius", "impressive", "perfect"
}

NEGATIVE_WORDS = {
    "hate", "bad", "terrible", "awful", "worst", "horrible", "sucks",
    "useless", "disappointing", "waste", "broken", "crash", "fail",
    "issue", "problem", "annoying", "frustrating", "stupid"
}