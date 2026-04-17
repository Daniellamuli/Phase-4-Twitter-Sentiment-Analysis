# src/clean_text.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.constants import *

# -----------------------------------
# Initialize NLP tools
# -----------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------------
# Remove URLs
# -----------------------------------
def remove_urls(text):
    if REMOVE_URLS:
        return re.sub(r"http\S+|www\S+", "", text)
    return text


# -----------------------------------
# Remove @mentions
# -----------------------------------
def remove_mentions(text):
    if REMOVE_MENTIONS:
        return re.sub(r"@\w+", "", text)
    return text


# -----------------------------------
# Remove hashtag symbol (#) but keep word
# -----------------------------------
def clean_hashtags(text):
    if REMOVE_HASHTAG_SYMBOLS:
        return text.replace("#", "")
    return text


# -----------------------------------
# Remove numbers
# -----------------------------------
def remove_numbers(text):
    if REMOVE_NUMBERS:
        return re.sub(r"\d+", "", text)
    return text


# -----------------------------------
# Remove punctuation
# -----------------------------------
def remove_punctuation(text):
    if REMOVE_PUNCTUATION:
        return re.sub(r"[^\w\s]", "", text)
    return text


# -----------------------------------
# Normalize whitespace
# -----------------------------------
def normalize_whitespace(text):
    return " ".join(text.split())


# -----------------------------------
# Remove stopwords
# -----------------------------------
def remove_stopwords(text):
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered)


# -----------------------------------
# Lemmatization
# -----------------------------------
def lemmatize_text(text):
    if USE_LEMMATIZATION:
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    return text


# -----------------------------------
# MAIN CLEANING FUNCTION
# -----------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = clean_hashtags(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)

    return text


# -----------------------------------
# APPLY CLEANING TO DATAFRAME
# -----------------------------------
def clean_dataframe(df):
    df[COL_CLEANED] = df[COL_TWEET].apply(clean_text)
    return df


# -----------------------------------
# TEST BLOCK (OPTIONAL)
# -----------------------------------
if __name__ == "__main__":
    sample = "I LOVE this iPhone!!! 😍 Visit https://apple.com #Amazing @user123"
    print("Original:", sample)
    print("Cleaned:", clean_text(sample))
