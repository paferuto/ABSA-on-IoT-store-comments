import streamlit as st
import pandas as pd
import numpy as np
# import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# load the data from pickle file
with open('tfidf.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
with open('LogisticAspect2.pkl','rb') as file2:
    logistic_aspect = pickle.load(file2)
with open('LogisticPolarity.pkl','rb') as file3:
    logistic_polarity = pickle.load(file3)

# Define preprocessing function
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import string

def preprocess(text):
    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove emoji
    # text = emoji.get_emoji_regexp().sub(u'', text)

    # Remove emoticon
    text = re.sub(r'\:\)|\:\(|\:\-\)|\:\-\(', '', text)

    # Remove comma
    text = re.sub(r',', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert n't to not
    text = re.sub(r'n\'t', 'not', text)

    # Tokenize text into words
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english')) - set(['not'])
    words = [w for w in words if not w in stop_words]

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w, pos='v') for w in words]

    # Join words back into text
    text = ' '.join(words)

    return text



def classify_sentiment(sentence, aspect_model, polarity_model, tfidf_vectorizer):
    # Split the input sentence into individual sentences using the full stop character
    comment = preprocess(sentence)
    # sentences = sentence.split('.')
    

    # Vectorize the preprocessed sentences
    X = tfidf_vectorizer.transform([comment])

    # Predict the aspects for each sentence using the aspect model
    aspect_preds = aspect_model.predict(X)

    # Classify the polarity of each sentence using the polarity model
    polarity_preds = polarity_model.predict(X)
    aspect =""
    aspect_pred = aspect_preds[0]
    if np.any(aspect_pred[0]==1): aspect+= " | Data integration"
    if np.any(aspect_pred[1]==1): aspect+= " | Marketing and Communication"
    if np.any(aspect_pred[2]==1): aspect+= " | Technology"
    if np.any(aspect_pred[3]==1): aspect+= " | Payment and Check-out"
    if np.any(aspect_pred[4]==1): aspect+= " | Shopping experience"
    if np.any(aspect_pred[5]==1): aspect+= " | Unemployment"
    if np.any(aspect_pred[6]==1): aspect+= " | Product availability and Store design"
    if np.any(aspect_pred[7]==1): aspect+= " | Price and Value"
    if np.any(aspect_pred[8]==1): aspect+= " | General"
    if np.any(aspect_pred[9]==1): aspect+= " | Privacy and Security"
    if(aspect!=""): aspect = aspect[3:]

    polarity = str(polarity_preds)
    polarity = preprocess(polarity)
    return aspect, polarity

# Create Streamlit app
st.title("Sentiment Analysis")
st.write("Enter text to classify sentiment:")

text = st.text_area("Text", height=200)

if st.button("Classify"):
    aspect, polarity = classify_sentiment(text,logistic_aspect,logistic_polarity,tfidf_vectorizer)
    st.write("Sentiment: ", polarity)
    st.write("Aspect: ", aspect)
