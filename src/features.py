import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# src/features.py

import re
import pandas as pd
from src.constants import COL_TWEET


# ============================================
# BASIC LENGTH FEATURES
# ============================================

def add_length_features(df):
    df['char_count'] = df[COL_TWEET].apply(lambda x: len(str(x)))
    df['word_count'] = df[COL_TWEET].apply(lambda x: len(str(x).split()))
    return df


# ============================================
# MENTION / HASHTAG / URL COUNTS
# ============================================

def add_social_features(df):
    df['mention_count'] = df[COL_TWEET].apply(lambda x: len(re.findall(r'@\w+', str(x))))
    df['hashtag_count'] = df[COL_TWEET].apply(lambda x: len(re.findall(r'#\w+', str(x))))
    df['url_count'] = df[COL_TWEET].apply(lambda x: len(re.findall(r'http\S+|www\S+', str(x))))
    return df


# ============================================
# PUNCTUATION + CAPITAL FEATURES
# ============================================

def add_text_style_features(df):
    df['exclamation_count'] = df[COL_TWEET].apply(lambda x: str(x).count('!'))
    df['question_count'] = df[COL_TWEET].apply(lambda x: str(x).count('?'))

    def capital_ratio(text):
        text = str(text)
        if len(text) == 0:
            return 0
        caps = sum(1 for c in text if c.isupper())
        return caps / len(text)

    df['capital_ratio'] = df[COL_TWEET].apply(capital_ratio)

    return df


# ============================================
# SENTIMENT WORD FEATURES (EXPANDED)
# ============================================

POSITIVE_WORDS = {
    "good", "great", "love", "loved", "loving", "awesome", "amazing",
    "fantastic", "excellent", "best", "perfect", "nice", "cool",
    "wonderful", "brilliant", "outstanding", "superb", "positive",
    "happy", "satisfied", "pleased", "enjoy", "enjoyed", "enjoying",
    "delightful", "impressive", "incredible", "fabulous", "terrific",
    "favorite", "favourite", "like", "liked", "likes"
}

NEGATIVE_WORDS = {
    "bad", "worst", "hate", "hated", "hating", "awful", "terrible",
    "horrible", "poor", "disappointing", "disappointed",
    "boring", "annoying", "frustrating", "negative",
    "sad", "angry", "upset", "problem", "issue", "bug",
    "fail", "failed", "failing", "broken", "slow", "lag",
    "crash", "crashes", "crashed", "error", "useless",
    "ridiculous", "pathetic", "waste", "disaster"
}


def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def add_sentiment_word_features(df):

    def count_positive(text):
        words = clean_and_tokenize(text)
        return sum(1 for w in words if w in POSITIVE_WORDS)

    def count_negative(text):
        words = clean_and_tokenize(text)
        return sum(1 for w in words if w in NEGATIVE_WORDS)

    df['positive_word_count'] = df[COL_TWEET].apply(count_positive)
    df['negative_word_count'] = df[COL_TWEET].apply(count_negative)

    return df


# ============================================
# COMPLETE FEATURE PIPELINE
# ============================================

def add_all_features(df):
    print("Adding engineered features...")

    df = add_length_features(df)
    df = add_social_features(df)
    df = add_text_style_features(df)
    df = add_sentiment_word_features(df)

    print("Feature engineering complete!")
    return df


# ============================================
# SIMPLE TEST 
# ============================================

if __name__ == "__main__":
    print("Features module loaded successfully!")
