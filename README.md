# Sentiment Analysis of Apple vs Google (SXSW Tweets)

## Team Members
- Daniella Muli
- Eve Michelle
- Naomi Opiyo
- Pheonverah Achieng'

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

---

