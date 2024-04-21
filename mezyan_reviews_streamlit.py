import numpy as np
import pandas as pd
import re
import nltk
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from bertopic import BERTopic
from umap import UMAP
import streamlit as st
#from selenium import webdriver
#from bs4 import BeautifulSoup
#import csv
#import time





all_reviews = pd.read_csv('all_reviews.csv', encoding='latin1')
stop_wordss = set(stopwords.words('english'))

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

    # Joining the tokens back into a string
    text = ' '.join(final_tokens)

    return text

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
    negation_words = {'not', 'no', 'never'}
    stop_words = stop_wordss - negation_words
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

    # Joining the tokens back into a string
    text = ' '.join(final_tokens)

    return text

all_reviews = all_reviews.dropna()
preprocess = FunctionTransformer(lambda X: X.apply(preprocess_text_2), validate=False)

RF_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('tfidf', TfidfVectorizer()),
    ('RF', RandomForestClassifier(max_depth=500, max_features='log2', n_estimators=100))
])

# Fit the pipeline on your data
X = all_reviews['Review Text']
y = all_reviews['Sentiment']
RF_pipeline.fit(X, y)

# Filter only the negative reviews
negative_reviews = all_reviews[all_reviews['Sentiment'] == 0]

negative_reviews['Processed Review Text'] = negative_reviews['Review Text'].apply(preprocess_text_1)
selected_columns = ['Review Name', 'Review Text', 'Processed Review Text', 'Sentiment']
negative_reviews = negative_reviews[selected_columns]

# Setting random_state in UMAP for reproducibility
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

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

# # Display the reviews for each topic
# for topic, reviews in reviews_by_topic.items():
#     print(f"Topic {topic}:")
#     for review in reviews:
#         print(f"- {review}")
#     print("\n")

topic_names = {
    0:"Food, Drinks, & General",
    1:"Food, Drinks, & General",
    2:"Food, Drinks, & General",
    3:"Service Quality & Staff",
    4:"Service Quality & Staff",
    5:"Service Quality & Staff",
    6:"Food, Drinks, & General",
    7:"Food, Drinks, & General",
    8:"Ambiance & Music",
    9:"Physical Setting & Hygiene",
    10:"Physical Setting & Hygiene",
    11:"Drinks",
    -1:"General"
}

# Add these names to your DataFrame
review_topics_df['Topic Name'] = review_topics_df['Topic'].map(topic_names)
# review_topics_df

# pd.set_option('display.max_colwidth', None)

# sorted_reviews = negative_reviews.sort_values(by='Review Text', key=lambda x: x.str.len(), ascending=False)
# top_10_reviews = sorted_reviews.head(5)

def analyze_text(text):
    # Convert text to a Series
    text_series = pd.Series([text])
    processed_text = preprocess_text_1(text_series.iloc[0])
    
    # Predict sentiment
    predicted_sentiment = RF_pipeline.predict(text_series)[0]

    # Process the text for topic modeling
    processed_text_series = pd.Series([processed_text])
    topics, _ = topic_model.transform(processed_text_series)
    topic_name = topic_names.get(topics[0], "Unknown Topic")
    
    return predicted_sentiment, topic_name

def process_csv(file):
    try:
        df = pd.read_csv(file, encoding='latin 1')
        df = df.dropna()
        if 'Reviewer Name' not in df.columns or 'Review Text' not in df.columns:
            return None, "CSV file must contain exactly two columns named 'Reviewer Name' and 'Review Text'."
        df['Sentiment'] = RF_pipeline.predict(df['Review Text'])
        df['Processed Review Text'] = df['Review Text'].apply(preprocess_text_1)
        processed_texts = df['Processed Review Text'].tolist()
        topics, _ = topic_model.transform(processed_texts)
        df['Topic'] = [topic_names.get(topic, "Unknown Topic") for topic in topics]
        df = df.drop('Processed Review Text', axis=1)
        return df, None
    except Exception as e:
        return None, str(e)

topic_names = {
    0:"Food, Drinks, & General",
    1:"Food, Drinks, & General",
    2:"Food, Drinks, & General",
    3:"Service Quality & Staff",
    4:"Service Quality & Staff",
    5:"Service Quality & Staff",
    6:"Food, Drinks, & General",
    7:"Food, Drinks, & General",
    8:"Ambiance & Music",
    9:"Physical Setting & Hygiene",
    10:"Physical Setting & Hygiene",
    11:"Drinks",
    -1:"General"
}



# def scrape_google_maps_reviews(url, duration):
#     # Setup Chrome options
#     chrome_options = webdriver.ChromeOptions()
#     chrome_options.add_argument("--lang=en-GB")

#     try:
#         driver = webdriver.Chrome(options=chrome_options)
#     except SessionNotCreatedException:
#         print("Error: Chrome driver (v114) and Chrome version installed on this computer are different. Please make sure that Chrome version is v114.")
#         return

#     # Open the URL
#     driver.get(url)

#     # Wait for the page elements to load
#     wait = WebDriverWait(driver, 10)
#     try:
#         menu_bt = wait.until(EC.element_to_be_clickable((By.XPATH, '//button[@data-value=\'Sort\']')))
#         menu_bt.click()
#     except Exception as e:
#         print("Error finding or clicking 'Sort' button:", e)
#         driver.quit()
#         return

#     time.sleep(5)

#     # Calculate the number of presses needed
#     interval = 1    # interval in seconds between each press
#     num_presses = duration // interval

#     # Perform the sequence of pressing End key
#     for _ in range(num_presses):
#         action = ActionChains(driver)
#         action.send_keys(Keys.END).perform()
#         time.sleep(interval)

#     time.sleep(5)  # Wait an additional 5 seconds after the loop

#     # Find the 'More' buttons and click them
#     more_buttons = driver.find_elements(By.XPATH, '//button[text()="More"]')
#     for button in more_buttons:
#         try:
#             button.click()
#             time.sleep(2)  # Small delay to allow content to load
#         except Exception as e:
#             print("Error clicking 'More' button:", e)

#     # Process and store reviews
#     response = BeautifulSoup(driver.page_source, 'html.parser')
#     review_elements = response.find_all('div', class_='jftiEf')
#     reviews = get_review_summary(review_elements)

#     # Save reviews to a CSV file
#     reviews_scraped = 'reviews.csv'
#     with open(reviews_scraped, 'w', newline='', encoding='utf-8') as file:
#         writer = csv.DictWriter(file, fieldnames=['Review Name', 'Review Rating', 'Review Text'])
#         writer.writeheader()
#         for review in reviews:
#             writer.writerow(review)

#     # Quit
#     driver.quit()

pd.set_option('display.max_colwidth', None)
sorted_reviews = all_reviews.sort_values(by='Review Text', key=lambda x: x.str.len(), ascending=False)



# Function to create tabs for Page 1
def page1():
    st.header("Mezyan")  # Heading for Page 1
    tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
    with tab1:
        st.write("Content of Page 1, Tab 1")
    with tab2:
        st.write("Content of Page 1, Tab 2")

# Function to create tabs for Page 2
def page2():
    st.header("Automatic Web Scraper from Google Maps Review Page")  # Heading for Page 2
    st.write("")
    url = st.text_input('Enter the Google Maps URL:')
    duration = st.number_input('Enter duration for scraping (in seconds):', min_value=5, max_value=600, value=30)
    st.markdown("""
    <style>
    .small-font {
        font-size:12px;
        color: red;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="small-font">Duration is based on the internet speed available<br>For a decent internet service a good duation would be equal to the number of reviews x 0.75</p>', unsafe_allow_html=True)        
    st.write("")

    st.write("Due to the limited resources available on Streamlit, this service has been disabled")
    
    # if st.button('Scrape Reviews'):
    #     if url:
    #         reviews = scrape_google_maps_reviews(url, duration)
    #         file_path = save_reviews(reviews)
    #         st.success('Scraping done! Download your file below.')
    #         with open(file_path, "rb") as file:
    #             btn = st.download_button(
    #                 label="Download CSV",
    #                 data=file,
    #                 file_name="reviews.csv",
    #                 mime="text/csv",
    #             )
    #     else:
    #         st.error('Please enter a valid URL.')    
        

# Function to create tabs for Page 3
def page3():
    st.header("Customer Feedback")  # Heading for Page 3
    tab1, tab2, tab3 = st.tabs(["Review Analysis","New Review", "New Bulk Reviews via CSV"])
    with tab1:
        # User selects sentiment type
        sentiment_type = st.radio("Choose Sentiment Type:", ('Positive', 'Negative'))
        sentiment_value = 1 if sentiment_type == 'Positive' else 0
        
        # Allow users to input a custom number of reviews to view
        num_reviews = st.number_input("Set the number of Reviews to View:", min_value=1, value=5, step=1)
        
        # Filter button
        if st.button('Filter Reviews'):
            # Ensure the DataFrame filtering does not exceed available reviews
            max_reviews = len(all_reviews[sorted_reviews['Sentiment'] == sentiment_value])
            num_reviews = min(num_reviews, max_reviews)
        
            # Filter the DataFrame based on the user's choice
            filtered_reviews = all_reviews[all_reviews['Sentiment'] == sentiment_value][:num_reviews]
        
            # Display the reviews
            if not filtered_reviews.empty:
                st.write(f"Showing top {num_reviews} {sentiment_type.lower()} reviews:")
                for index, row in filtered_reviews.iterrows():
                    st.write(f"Review {index + 1}: {row['Review Text']}")
            else:
                st.write("No reviews to display.")     
        

    with tab2:
        st.header("Analyze a Single Review")
        name = st.text_input("Name")
        review = st.text_area("Review")
        st.markdown("""
        <style>
        .small-font {
            font-size:12px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Using <br> tag to create a new line in HTML
        st.markdown('<p class="small-font">Example:<br>I loved the mezza, very good food.<br>Bad. The food lacked flavor, drinks are watered down.</p>', unsafe_allow_html=True)        
        if st.button("Analyze Sentiment"):
            predicted_sentiment, topic_name = analyze_text(review)
            sentiment_label = 'Positive' if predicted_sentiment == 1 else 'Negative'
            st.write(f"{name} just left a {sentiment_label} review (Topic: {topic_name})")

    with tab3:
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

# Function to create tabs for Page 4
def page4():
    st.header("Restocking Chicken Breast")  # Heading for Page 4
    tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
    with tab1:
        st.write("Content of Page 4, Tab 1")
    with tab2:
        st.write("Content of Page 4, Tab 2")

# Main function to manage navigation and universal app title
def main():
    st.title("Mezyan - Analytical Approach")  # Universal title for the app

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a Page", ["Mezyan", "Web Scraper", "Customer Feedback", "Restocking Chicken Breast"])

    if page == "Mezyan":
        page1()
    elif page == "Web Scraper":
        page2()
    elif page == "Customer Feedback":
        page3()
    elif page == "Restocking Chicken Breast":
        page4()

if __name__ == "__main__":
    main()
