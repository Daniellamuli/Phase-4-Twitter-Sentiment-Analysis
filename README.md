<div align="center">
  <img src="figures/twitter-logo.jpg" width="200" height="200" style="border-radius:50%; display:block;">

# Sentiment Analysis of Apple vs Google (SXSW Tweets)

## Team Members
Daniella Muli • Eve Michelle • Naomi Opiyo • Pheonverah Achieng'
</div>

---

## Introduction

In the tech industry, public perception can shift almost instantly. A single tweet can influence how a product or brand is perceived at scale. This project explores how Natural Language Processing (NLP) can be used to understand and classify that perception.

Using a dataset of over 9,000 tweets collected during the SXSW (South by Southwest) conference, the goal is to analyze how people feel about two major technology companies: Apple and Google. The project begins with a simple binary classification approach and gradually expands into a multiclass model capable of identifying positive, negative, and neutral sentiment.

At its core, this work serves as a proof of concept for automated brand monitoring—demonstrating how machine learning can turn unstructured social media data into meaningful insights.

---

## Business Understanding

In a highly competitive tech landscape, companies like Apple and Google depend on real-time consumer feedback to stay ahead. Sentiment is more than just public opinion—it directly influences brand value, product success, and customer trust.

Social media platforms provide a constant stream of this feedback, especially during major events like SXSW, where conversations around products and brands spike significantly. In theory, this gives companies direct access to real-time customer opinions.

In practice, however, the volume and speed of this data make it difficult to extract meaningful insights. Thousands of tweets can appear within minutes, mixing valuable feedback with irrelevant or repetitive content. As a result, important signals—such as early dissatisfaction with a product or emerging praise—are easily buried in the noise.

At this scale, manual analysis becomes not just inefficient, but unreliable—slowing down response times and making it harder for teams to act with confidence.

### The Business Problem

These challenges translate into several practical limitations for organizations managing their digital reputation:

- **Scalability Gap**  
  The volume of social media data exceeds what teams can realistically monitor in real time, leading to missed insights.

- **Undetected Reputation Risks**  
  Negative sentiment can gain traction before teams are able to respond, potentially causing lasting brand damage.

- **Fragmented Insight & Comparison**  
  There is no consistent, objective way to track brand health or compare how consumers perceive Apple versus Google products.

- **Reactive vs Proactive Decision-Making**  
  Without structured sentiment data, teams are often forced into reactive responses instead of proactively managing brand perception.

### Objective

To address these challenges, this project develops a sentiment analysis pipeline that automates the classification of tweets. By transforming unstructured text into structured insights, the model enables organizations to:

- **Monitor Brand Health** in real time  
- **Detect negative sentiment early** to mitigate risks  
- **Identify positive sentiment** to understand what resonates with users  
- **Compare market perception** between competing products  

Ultimately, the goal is to turn large volumes of social media data into clear, actionable insights that support better decision-making in marketing, product development, and customer engagement.

---

## Data Understanding

The dataset used in this project was sourced from CrowdFlower and made available via data.world. It contains 9,093 tweets, each labeled by human annotators to reflect the sentiment expressed.

### Data Structure:

* **`tweet_text`**: The raw text content of the tweet.
* **`emotion_in_tweet_is_directed_at`**: The specific product or brand the emotion is targeting (e.g., iPad, iPhone, Google, Android).
* **`is_there_an_emotion_directed_at_a_brand_or_product`**: The target variable representing sentiment (e.g., positive, negative, neutral).

#### Sentiment Distribution
Sentiment 	   Count	         Percentage

- Positive	  2,978	           33.3%

- Neutral	    5,389	            60.3%

- Negative	    570	          6.4%


The dataset is heavily imbalanced, with negative sentiment representing only ~6% of labeled examples. This directly influenced our choice of F1 score over accuracy as the primary evaluation metric.

### Top Mentioned Products

![](figures/top_10_products.png)

**This chart shows the most frequently mentioned products and brands across all tweets.

| Rank | Product | Mention Count | Business Insight |
|------|---------|---------------|------------------|
| 1 | iPhone | Highest | Apple's flagship drives most conversation |
| 2 | iPad | Very High | Strong tablet discussions at SXSW |
| 3 | Google | High | Broad brand discussion (not product-specific) |
| 4 | Android | High | Primary competitor to iPhone |
| 5 | iPad or iPhone App | Medium | Developer ecosystem matters |

**Why this matters:**
 We focused our sentiment analysis on these top products since they represent the majority of conversations. Products with fewer mentions were grouped into broader categories during feature engineering.


### Data Preparation
The raw tweet dataset underwent a series of essential data preparation steps to ensure its quality and suitability for machine learning:

1. Text Cleaning Pipeline
Raw tweet text is extremely noisy. The following cleaning steps were applied in sequence:

Step	Operation	Rationale
- Lowercasing	Ensures consistent tokenization.
- URL removal	URLs carry no sentiment information
- 	Mention removal (@handles)	User handles are uninformative
- Hashtag symbol removal (#)	Retains the word (e.g., #amazing)
- Number removal	Digits don't contribute to sentiment
- Punctuation removal	Standardizes tokens.
- Stopword removal	Removes common words without signal.
- Lemmatisation	Reduces words to base form.

2. Label Encoding
Binary Classification: Positive (1) vs Negative (0) — neutral tweets excluded.

Multiclass Classification: Positive (2), Neutral (1), Negative(0).

3. Data Splitting
The dataset was split into training (80%) and test (20%) subsets using stratified sampling to maintain class proportions.

4. Feature Engineering

TF-IDF Vectorisation with:

max_features=5000 (top 5,000 most informative terms)
ngram_range=(1, 2) (unigrams + bigrams)


### Modeling
he goal was to predict sentiment from tweet text. Two classification tasks were addressed:

1. Binary Classification (Positive vs Negative)
Model	Description: Logistic Regression	Baseline model due to interpretability, uses L2 regularization.
Multinomial Naive Bayes	Probabilistic classifier designed for text data
2. Multiclass Classification (Positive vs Neutral vs Negative)
Random Forest	Ensemble model with 100 trees, max depth 10
Linear SVM	Optimized for high-dimensional sparse text data

### Model Evaluation
Model performance was primarily evaluated using the Weighted F1 Score due to class imbalance.

The Naive Bayes (Binary) model is the strongest and most reliable choice, delivering the best overall performance with an F1 score of 0.8086 and an accuracy of 0.8577. It not only outperforms all other models in the evaluation but also demonstrates a strong balance between precision (0.8685) and recall (0.8577), meaning it consistently makes accurate predictions while minimizing both false positives and false negatives. This makes it a dependable model for real-world deployment compared to Logistic Regression, SVM, and Random Forest.

![](figures/confusion_matrices.png)

True Positives (TP): 595 - Correctly predicted positive sentiment.

True Negatives (TN): 14 - Correctly predicted negative sentiment.

False Positives (FP): 100 - Incorrectly predicted as positive sentiment.

False Negatives (FN): 1 - Incorrectly predicted as negative sentiment.


## Key Findings

### 1. Words That Drive Sentiment

As shown in the word clouds below, certain words strongly correlate with sentiment:

- **Positive tweets** frequently contain: *love*, *amazing*, *great*, *awesome*
- **Negative tweets** frequently contain: *crash*, *dead*, *hate*, *terrible*

![](figures/wordcloud_positive.png)
![](figures/wordcloud_negative.png)

### 2. What We Learned

| Finding | Implication |
|---------|-------------|
| Negative sentiment is hardest to detect | Only ~6% of dataset — needs more examples |
| TF-IDF with unigrams + bigrams works | Captures sentiment-bearing phrases effectively |
| Words matter more than structure | Linguistic features > tweet length/metadata |
| Both success criteria met | Binary F1: 0.8086, Multiclass F1: 0.70 |

### 3. Linguistic Patterns That Matter

Across all models, four patterns consistently drove sentiment classification:

1. **Positive emotion words** (love, amazing, great)
2. **Negative problem indicators** (crash, dead, hate)
3. **Punctuation patterns** (! for excitement, ? for complaints)
4. **Informational language** (neutral, factual statements)

### Conclusion
 Sentiment is expressed more through **word choice and tone** than tweet structure alone.

The **Naive Bayes (Binary) model** achieved the best performance with an **F1 Score of 0.8086** (exceeding our 0.80 target) and **85.8% accuracy**.

### Recommendations
Based on key linguistic patterns the following were identified:

- Monitor Positive Keywords	Track words like "love," "great," "amazing" as early indicators of campaign success
- Flag Negative Keywords	Prioritize tweets containing "crash," "dead," "battery" for immediate support response.
- Track Sentiment Trends	Monitor sentiment before and after product launches.
- Identify Brand Advocates	Engage users who consistently post positive content.
- Route Negative Tweets	Automatically escalate negative tweets to support channels.
- Apply Confidence Threshold - Use threshold of 0.7 to balance automation with prediction reliability.

## Next Steps
To enhance the impact of this project:

Priority	Task	Description
- High	Transformer Models	- Implement BERT/RoBERTa for better context and sarcasm detection
- High	Real-Time Pipeline	Integrate with Twitter/X API for live sentiment monitoring
- Medium	Interactive Dashboard	Enhance Tableau dashboard with live data feeds
- Medium	Multilingual Support - xtend analysis to additional languages
- Low	Aspect-Based Sentiment - Analyze sentiment toward specific product features
- Low	Emoji Mapping	- Add emoji-to-sentiment mapping instead of removal

Stakeholder Collaboration

Effective implementation of these insights requires collaboration between:


- Marketing Teams: Use sentiment trends to measure campaign effectiveness.
- Product Teams:	Prioritize issues highlighted in negative tweets
- Customer Support Teams: Automatically route negative tweets for rapid response
- Data Science Teams: Maintain and improve models with new data
- By integrating predictive analytics into brand management strategies, organizations can design more targeted and effective reputation management campaigns.


# Setup Instructions
bash
## 1. Clone the repository
git clone https://github.com/your-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

## 2. Create virtual environment
python -m venv venv

## 3. Activate virtual environment
### Windows:
venv\Scripts\activate
### macOS/Linux:
source venv/bin/activate

## 4. Install dependencies
pip install -r requirements.txt

## 5. Download NLTK data (automatically handled in notebook)

## 6. Launch Jupyter Notebook
jupyter notebook

## Requirements.txt
text
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
wordcloud
jupyter

## Project Structure


```
twitter-sentiment-analysis/
|
|-- data/
|   `-- judge-1377884607_tweet_product_company.csv
|
|-- figures/
|   |-- feature_correlation_heatmap.png
|   |-- model_comparison.png
|   |-- sentiment_distribution.png
|   |-- top_10_products.png
|   |-- twitter-logo.jpg
|   |-- wordcloud_all.png
|   |-- wordcloud_negative.png
|   `-- wordcloud_positive.png
|
|-- notebooks/
|   `-- final_notebook.ipynb
|
|-- presentation/
|   `-- Twitter Sentiment Analysis.pdf
|
|-- src/
|   |-- __init__.py
|   |-- clean_text.py
|   |-- compare_models.py
|   |-- constants.py
|   |-- evaluate.py
|   |-- export_tableau_data.py
|   |-- features.py
|   |-- load_data.py
|   |-- main.py
|   |-- pipeline.py
|   |-- preprocess.py
|   |-- split_data.py
|   |-- train_binary.py
|   |-- train_multiclass.py
|   |-- vectorize.py
|   `-- visualize.py
|
|-- tableau/
|
|-- .gitignore
|-- PROJECT_PLAN.md
|-- README.md
`-- requirements.txt



 ## Project Structure

```twitter-sentiment-analysis/
│
├── data/
│   └── judge-1377884607_tweet_product_company.csv
│
├── figures/
│   ├── feature_correlation_heatmap.png
│   ├── model_comparison.png
│   ├── sentiment_distribution.png
│   ├── top_10_products.png
│   ├── twitter-logo.jpg
│   ├── wordcloud_all.png
│   ├── wordcloud_negative.png
│   └── wordcloud_positive.png
│
├── notebooks/
│   └── final_notebook.ipynb
│
├── presentation/
│   └── Twitter Sentiment Analysis.pdf
│
├── src/
│   ├── __init__.py
│   ├── clean_text.py
│   ├── compare_models.py
│   ├── constants.py
│   ├── evaluate.py
│   ├── export_tableau_data.py
│   ├── features.py
│   ├── load_data.py
│   ├── main.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── split_data.py
│   ├── train_binary.py
│   ├── train_multiclass.py
│   ├── vectorize.py
│   └── visualize.py
│
├── tableau/
│
├── .gitignore
├── PROJECT_PLAN.md
├── README.md
└── requirements.txt

```

###### Team Structure

- Daniella Muli (Lead)	- Preprocessing pipeline, binary classification, final notebook.
- Eve Michelle	- Data ingestion, text vectorisation (TF-IDF), multiclass modelling.
- Naomi Opiyo	- Exploratory data analysis, data splitting, model comparison.
- Pheonverah Achieng'	- Text cleaning, feature engineering, model evaluation.


---

## Live Interactive Dashboard

To make our results accessible to non-technical stakeholders, we created an interactive Tableau dashboard that visualizes the key findings from our sentiment analysis.

**Click the link below to explore the dashboard:**https://public.tableau.com/views/SXSWTwitterSentimentAnalysis/Dashboard1?:language=en-GB&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link



##### Acknowledgments
- CrowdFlower (now Figure Eight) for providing the annotated dataset.
- data.world for hosting the SXSW Tweet Sentiment dataset
- The open-source Python community for the amazing libraries used

##### License
This project is licensed under the MIT License.


