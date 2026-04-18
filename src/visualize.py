#src/visualize.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
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

PLOTS_DIR = "figures"
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



##VISUALIZATION
##TOP 10 MENTIONED PRODUCTS/BRANDS

def analyze_top_products(df):
    top_products = df[COL_PRODUCT].value_counts().head(10)
 
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)
 
    top_products.sort_values().plot(kind="barh", color=COLOR_NEUTRAL, edgecolor="white", ax=ax)
 
    ax.set_title("Top 10 Most Mentioned Products / Brands", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Mentions")
    ax.set_ylabel("Product / Brand")
 
    for bar in ax.patches:
        ax.text(
            bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2,
            str(int(bar.get_width())),
            va="center", fontsize=9
        )
 
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_10_products.png"), dpi=150)
    plt.close()

## SENTIMENT DISTRIBUTION BAR CHART

def plot_sentiment_distribution(df):
    sentiment_order = [POSITIVE, NEUTRAL, NEGATIVE]
    color_map = {POSITIVE: COLOR_POSITIVE, NEUTRAL: COLOR_NEUTRAL, NEGATIVE: COLOR_NEGATIVE}
 
    counts = df[COL_SENTIMENT].value_counts().reindex(sentiment_order).dropna()
    colors = [color_map[s] for s in counts.index]
 
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)
 
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", width=0.5)
 
    for bar, count in zip(bars, counts.values):
        pct = (count / counts.sum()) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{count} ({pct:.1f}%)",
            ha="center", fontsize=10, fontweight="bold"
        )
 
    ax.set_title("Sentiment Distribution of SXSW Tweets", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Tweets")
    ax.set_xticklabels(["Positive", "Neutral", "Negative"])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
 
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sentiment_distribution.png"), dpi=150)
    plt.close()

##WORD CLOUDS

def generate_word_clouds(df):
    text_col = COL_CLEANED if COL_CLEANED in df.columns else COL_TWEET
 
    subsets = {
        "all":      (df,                                 "All Tweets",      "RdYlGn"),
        "positive": (df[df[COL_SENTIMENT] == POSITIVE], "Positive Tweets", "Greens"),
        "negative": (df[df[COL_SENTIMENT] == NEGATIVE], "Negative Tweets", "Reds"),
    }
 
    for key, (subset, title, colormap) in subsets.items():
        text = " ".join(subset[text_col].dropna().values)
 
        wc = WordCloud(
            width=WORDCLOUD_WIDTH,
            height=WORDCLOUD_HEIGHT,
            max_words=WORDCLOUD_MAX_WORDS,
            background_color=WORDCLOUD_BACKGROUND,
            colormap=colormap,
        ).generate(text)
 
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(COLOR_BACKGROUND)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud - {title}", fontsize=14, fontweight="bold")
 
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"wordcloud_{key}.png"), dpi=150)
        plt.close()


##COMPLETE PIPELINE 

def run_visualization_pipeline(df):
    document_data_dictionary(df)
    analyze_top_products(df)
    plot_sentiment_distribution(df)
    generate_word_clouds(df)
 
 
if __name__ == "__main__":
    from src.load_data import load_and_prepare_data
    from src.clean_text import clean_dataframe
 
    df = load_and_prepare_data()
    df = clean_dataframe(df)
    run_visualization_pipeline(df)