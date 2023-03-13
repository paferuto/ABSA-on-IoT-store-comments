import streamlit as st
import pandas as pd
import numpy as np
from preprocess import preprocess
# import joblib
import pickle
# Define preprocessing function
import re
import nltk
import nltk
pd.options.mode.chained_assignment = None


# load the data from pickle file
with open('tfidf.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)
with open('LogisticAspect2.pkl','rb') as file2:
    logistic_aspect = pickle.load(file2)
with open('LogisticPolarity.pkl','rb') as file3:
    logistic_polarity = pickle.load(file3)



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import string


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
    if np.any(aspect_pred[1]==1): aspect+= " | Marketing, Communication & Special offers"
    if np.any(aspect_pred[2]==1): aspect+= " | Technology"
    if np.any(aspect_pred[3]==1): aspect+= " | Payment and Check-out"
    if np.any(aspect_pred[4]==1): aspect+= " | Shopping experience"
    if np.any(aspect_pred[5]==1): aspect+= " | Unemployment"
    if np.any(aspect_pred[6]==1): aspect+= " | Product availability and Store design"
    if np.any(aspect_pred[7]==1): aspect+= " | Price and Value"
    if np.any(aspect_pred[8]==1): aspect+= " | General"
    if np.any(aspect_pred[9]==1): aspect+= " | Privacy and Security issues"
    if(aspect!=""): aspect = aspect[3:]

    polarity = str(polarity_preds)
    polarity = preprocess(polarity)
    return aspect, polarity

# Create Streamlit app
st.title("Aspect-based sentiment analysis on IoT store comments")
st.write("Ứng dụng này được xây dựng dựa trên mô hình chúng tôi dùng để phân tích dữ liệu trong bài nghiên cứu. Dù chung tôi vẫn giữ nguyên framework như đã được trình bày ở quy trình nghiên cứu, một số thay đổi trong quy trình xử lý đã được thực hiện để việc xử lý phân loại các bình luận đơn lẻ được tiện lợi hơn. dịch đoạn này ra tiếng Anh")
st.write("Please input a short sample comment expressing your viewpoint on the experience at retail stores with IoT applications. Our model is built on English language data, so please enter an English comment to allow the model to classify it accurately.")
text = st.text_area("", height=200)

if st.button("Classify"):
    aspect, polarity = classify_sentiment(text,logistic_aspect,logistic_polarity,tfidf_vectorizer)
    st.write("Sentiment: ", polarity)
    st.write("Aspect: ", aspect)
