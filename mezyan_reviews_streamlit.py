# Import Libraries
import numpy as np
import pandas as pd
import re
import nltk
import contractions
import torch
import gensim.downloader as api
import random

# Import specific functions and classes
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from gensim.models import Word2Vec

# Download NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
from bertopic import BERTopic
import umap as UMAP
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)


import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import io



def preprocess_text_1(text):

    # Replace the backslash with out of (3/5 with 3 out of 5)
    text = re.sub(r'(\d+)/(\d+)', lambda m: f"{m.group(1)} out of {m.group(2)}", text)

    # Lowercase and remove special characters\whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s+*/.\-]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()

    # Remove punctuation, numbers(except for ratings), URLs, HTML tags
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text= re.sub(r'(?<!out of )\b\d+\b(?! out of)','', text)
    text = re.sub(r"\$", "USD", text)
    text = ' '.join(text.split())
    text = contractions.fix(text)


    # Function to combine 'out' and 'of' into 'out_of'
    def combine_out_of(tokens):
        combined_tokens = []
        skip_next = False
        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            if token == 'out' and i + 1 < len(tokens) and tokens[i + 1] == 'of':
                combined_tokens.append('out_of')
                skip_next = True
            else:
                combined_tokens.append(token)
        return combined_tokens

    # Tokenization
    tokens = word_tokenize(text)
    tokens = combine_out_of(tokens)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    final_tokens = []
    for token in lemmatized_tokens:
        if token == "out_of":
            final_tokens.extend(["out", "of"])
        else:
            final_tokens.append(token)

    # Part-of-Speech Tagging
    pos_tags = nltk.pos_tag(final_tokens)

    # Named Entity Recognition
    ner = nltk.ne_chunk(pos_tags)

    # Joining the tokens back into a string
    text = ' '.join(final_tokens) #, pos_tags, ner

    return text, pos_tags, ner

def preprocess_text_2(text):

    # Replace the backslash with out of (3/5 with 3 out of 5)
    text = re.sub(r'(\d+)/(\d+)', lambda m: f"{m.group(1)} out of {m.group(2)}", text)

    # Lowercase and remove special characters\whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s+*/.\-]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()

    # Remove punctuation, numbers(except for ratings), URLs, HTML tags
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text= re.sub(r'(?<!out of )\b\d+\b(?! out of)','', text)
    text = re.sub(r"\$", "USD", text)
    text = ' '.join(text.split())
    text = contractions.fix(text)


    # Function to combine 'out' and 'of' into 'out_of'
    def combine_out_of(tokens):
        combined_tokens = []
        skip_next = False
        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            if token == 'out' and i + 1 < len(tokens) and tokens[i + 1] == 'of':
                combined_tokens.append('out_of')
                skip_next = True
            else:
                combined_tokens.append(token)
        return combined_tokens

    # Tokenization
    tokens = word_tokenize(text)
    tokens = combine_out_of(tokens)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'never'}
    stop_words = stop_words - negation_words
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    final_tokens = []
    for token in lemmatized_tokens:
        if token == "out_of":
            final_tokens.extend(["out", "of"])
        else:
            final_tokens.append(token)

    # Part-of-Speech Tagging
    pos_tags = nltk.pos_tag(final_tokens)

    # Named Entity Recognition
    ner = nltk.ne_chunk(pos_tags)

    # Joining the tokens back into a string
    text = ' '.join(final_tokens) #, pos_tags, ner

    return text, pos_tags, ner

def map_sentiment_1(rating):
    if rating in [1, 2]:
        return 0  # Negative sentiment
    elif rating in [3, 4, 5]:
        return 1  # Positive sentiment

def map_sentiment_2(rating):
    if rating in [1, 2, 3]:
        return 0  # Negative sentiment
    elif rating in [4, 5]:
        return 1  # Positive sentiment

"""# Roberta"""

def predict_sentiment_with_probability(text):
    preprocessed_text, _, _ = preprocess_text_1(text)  #WITHOUT REMOVAL OF STOPWORDS
    encoded_input = tokenizer(preprocessed_text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    label = np.argmax(scores)
    probability_of_label_2 = scores[2]

    return label, probability_of_label_2

def align_sentiment(predicted_sentiment):
    # Map 0 and 1 to 0 (negative), and 2 to 1 (positive)
    if predicted_sentiment in [0, 1]:
        return 0
    elif predicted_sentiment == 2:
        return 1
    else:
        raise ValueError("Invalid sentiment value")

all_reviews = pd.read_csv('/content/drive/MyDrive/Capstone/all_reviews.csv', encoding='latin1')

# Filter only the negative reviews
negative_reviews = all_reviews[all_reviews['Sentiment'] == 0]
negative_reviews.shape

pd.set_option('display.max_colwidth', None)

sorted_reviews = negative_reviews.sort_values(by='Review Text', key=lambda x: x.str.len(), ascending=False)
top_10_reviews = sorted_reviews.head(5)

"""## Bert Topic Modeling"""

negative_reviews=negative_reviews.dropna()

negative_reviews['Processed Review Text'] = negative_reviews['Review Text'].apply(lambda x: preprocess_text_1(x)[0])
selected_columns = ['Review Name', 'Review Text', 'Processed Review Text', 'Sentiment']
negative_reviews = negative_reviews[selected_columns]

# Setting random_state in UMAP for reproducibility
umap_model = UMAP.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

# Instantiate BERTopic with the UMAP model
topic_model = BERTopic(umap_model=umap_model, n_gram_range=(3, 10), language="english")

# Fit the model
topics, probabilities = topic_model.fit_transform(negative_reviews['Processed Review Text'])
topic_model.visualize_topics()

# Create a DataFrame containing the reviews and their assigned topics
review_topics_df = pd.DataFrame({
    'Review': negative_reviews['Review Text'],
    'Topic': topics
})

# Initialize a dictionary to store the reviews by topic
reviews_by_topic = {}

# Group by 'Topic' and sample 5 reviews from each topic
for topic_number in set(topics):
    # Filter the DataFrame for the current topic
    topic_df = review_topics_df[review_topics_df['Topic'] == topic_number]

    # Check if there are at least 5 reviews
    n_samples = min(15, len(topic_df))

    # Sample reviews
    sampled_reviews = topic_df.sample(n=n_samples, random_state=42)

    # Store the sampled reviews in the dictionary
    reviews_by_topic[topic_number] = sampled_reviews['Review'].tolist()

# Display the reviews for each topic
for topic, reviews in reviews_by_topic.items():
    print(f"Topic {topic}:")
    for review in reviews:
        print(f"- {review}")
    print("\n")

topic_names = {
    0:"Food, Drinks, & General",
    1:"Food, Drinks, & General",
    2:"Food, Drinks, & General",
    3:"Ingredients Taste & Quality",
    4:"Service Quality & Staff",
    5:"Service Quality & Staff",
    6:"Service Quality & Staff",
    7:"Ambiance & Music",
    8:"Hygiene",
    9:"Physical Setting",
    10:"Food, Drinks, & General",
    11:"Food, Drinks, & General",
    -1:"Food, Drinks, & General"
}

# Add these names to your DataFrame
review_topics_df['Topic Name'] = review_topics_df['Topic'].map(topic_names)

review_topics_df

"""# Streamlit App"""

def preprocess_text(text):

    # Replace the backslash with out of (3/5 with 3 out of 5)
    text = re.sub(r'(\d+)/(\d+)', lambda m: f"{m.group(1)} out of {m.group(2)}", text)

    # Lowercase and remove special characters\whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s+*/.\-]', '', text, re.I | re.A)
    text = text.lower()
    text = text.strip()

    # Remove punctuation, numbers(except for ratings), URLs, HTML tags
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text= re.sub(r'(?<!out of )\b\d+\b(?! out of)','', text)
    text = re.sub(r"\$", "USD", text)
    text = ' '.join(text.split())
    text = contractions.fix(text)


    # Function to combine 'out' and 'of' into 'out_of'
    def combine_out_of(tokens):
        combined_tokens = []
        skip_next = False
        for i, token in enumerate(tokens):
            if skip_next:
                skip_next = False
                continue
            if token == 'out' and i + 1 < len(tokens) and tokens[i + 1] == 'of':
                combined_tokens.append('out_of')
                skip_next = True
            else:
                combined_tokens.append(token)
        return combined_tokens

    # Tokenization
    tokens = word_tokenize(text)
    tokens = combine_out_of(tokens)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    final_tokens = []
    for token in lemmatized_tokens:
        if token == "out_of":
            final_tokens.extend(["out", "of"])
        else:
            final_tokens.append(token)

    # Joining the tokens back into a string
    text = ' '.join(final_tokens)

    return text

def predict_sentiment(text):
    preprocessed_text, _, _ = preprocess_text_1(text)  #WITHOUT REMOVAL OF STOPWORDS
    encoded_input = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores.argmax()

def align_sentiment(predicted_sentiment):
    if predicted_sentiment in [0, 1]:
        return 0
    elif predicted_sentiment == 2:
        return 1
    else:
        raise ValueError("Invalid sentiment value")

def analyze_text(text):
    processed_text = preprocess_text(text)
    sentiment = predict_sentiment_with_probability(processed_text)
    aligned_sentiment = align_sentiment(sentiment)
    topics, _ = topic_model.fit_transform([processed_text])
    topic_name = topic_names.get(topics[0], "Unknown Topic")
    return aligned_sentiment, topic_name

def process_csv(file):
    try:
        df = pd.read_csv(file)
        if 'Reviewer Name' not in df.columns or 'Review Text' not in df.columns:
            return None, "CSV file must contain exactly two columns named 'Reviewer Name' and 'Review Text'."
        df['Processed Text'] = df['Review Text'].apply(preprocess_text)
        df['Sentiment'] = df['Processed Text'].apply(predict_sentiment_with_probability).apply(align_sentiment)
        topics, _ = topic_model.fit_transform(df['Processed Text'].tolist())
        df['Topic'] = [topic_names.get(topic, "Unknown Topic") for topic in topics]
        return df, None
    except Exception as e:
        return None, str(e)

topic_names = {
    0:"Food, Drinks, & General",
    1:"Food, Drinks, & General",
    2:"Food, Drinks, & General",
    3:"Ingredients Taste & Quality",
    4:"Service Quality & Staff",
    5:"Service Quality & Staff",
    6:"Service Quality & Staff",
    7:"Ambiance & Music",
    8:"Hygiene",
    9:"Physical Setting",
    10:"Food, Drinks, & General",
    11:"Food, Drinks, & General",
    -1:"Food, Drinks, & General"
}

st.title('Sentiment Analysis App')

tab1, tab2 = st.tabs(["Single Review", "Bulk Reviews via CSV"])

with tab1:
    st.header("Analyze a Single Review")
    name = st.text_input("Name")
    review = st.text_area("Review")
    if st.button("Analyze Sentiment"):
        predicted_sentiment = predict_sentiment_with_probability(review)
        aligned_sentiment = align_sentiment(predicted_sentiment)
        sentiment_label = 'Positive' if aligned_sentiment == 1 else 'Negative'
        st.write(f"""{name} just left a {sentiment_label} review
        {review}""")

with tab2:
    st.header("Upload CSV for Bulk Analysis")
    st.markdown("Please upload a CSV file with two columns: 'Reviewer Name' and 'Review Text'")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        data, error = process_csv(uploaded_file)
        if error:
            st.error(f"Error: {error}")
        elif data is not None:
            st.write("Analysis Complete. Here are the results:")
            st.dataframe(data[['Reviewer Name', 'Review Text', 'Sentiment', 'Topic']])
            # Convert DataFrame to CSV for download
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download processed CSV",
                data=csv,
                file_name='processed_reviews.csv',
                mime='text/csv',
            )
