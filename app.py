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


# def classify_sentiment(sentence, aspect_model, polarity_model, tfidf_vectorizer):
#     # Split the input sentence into individual sentences using the full stop character
#     comment = preprocess(sentence)
#     # sentences = sentence.split('.')
    

#     # Vectorize the preprocessed sentences
#     X = tfidf_vectorizer.transform([comment])

#     # Predict the aspects for each sentence using the aspect model
#     aspect_preds = aspect_model.predict(X)

#     # Classify the polarity of each sentence using the polarity model
#     polarity_preds = polarity_model.predict(X)
#     aspect =""
#     aspect_pred = aspect_preds[0]
#     if np.any(aspect_pred[0]==1): aspect+= " | Data integration"
#     if np.any(aspect_pred[1]==1): aspect+= " | Marketing, Communication & Special offers"
#     if np.any(aspect_pred[2]==1): aspect+= " | Technology"
#     if np.any(aspect_pred[3]==1): aspect+= " | Payment and Check-out"
#     if np.any(aspect_pred[4]==1): aspect+= " | Shopping experience"
#     if np.any(aspect_pred[5]==1): aspect+= " | Unemployment"
#     if np.any(aspect_pred[6]==1): aspect+= " | Product availability and Store design"
#     if np.any(aspect_pred[7]==1): aspect+= " | Price and Value"
#     if np.any(aspect_pred[8]==1): aspect+= " | General"
#     if np.any(aspect_pred[9]==1): aspect+= " | Privacy and Security issues"
#     if(aspect!=""): aspect = aspect[3:]

#     polarity = str(polarity_preds)
#     polarity = preprocess(polarity)
#     if(polarity=="positi"):polarity="Positive"
#     elif(polarity=="negati"):polarity="Negative"
#     elif(polarity=="neutral"): polarity="Neutral"
#     return aspect, polarity

def predict_aspect(text):
    aspect_preds = logistic_aspect.predict(text)
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
    return aspect


def predict_polarity(vec):
    polarity_preds = logistic_polarity.predict(vec)
    polarity = str(polarity_preds)
    polarity = preprocess(polarity)
    if(polarity=="positi"):polarity="Positive"
    elif(polarity=="negati"):polarity="Negative"
    elif(polarity=="neutral"): polarity="Neutral"
    return polarity

def predict_multi_aspect(text):
    # Tách câu dài hoặc đoạn nhiều câu bằng các từ như "but", "except", "yet", dấu chấm
    sentences = re.split(r'[.!?;]|but |except |yet |although |though |however |despite |or |nor', text)
    
    aspects = ""
    polarities = ""
    aspectArray=[]
    # Tiền xử lý và dự đoán các khía cạnh trong từng câu
    for sentence in sentences:
        # Tiền xử lý văn bản
        preprocessed = preprocess(sentence)
        aspect =""
        polarity=""
        if(len(preprocessed)>2):
            vec = tfidf_vectorizer.transform([preprocessed])
            aspect = predict_aspect(vec)
            polarity = predict_polarity(vec)
        # Dự đoán các khía cạnh trong câu
        if aspect !="" and aspect not in aspectArray:
            aspects+= " | "+aspect
            aspectArray.append(aspect)
        if polarity !="": polarities+=" | "+polarity
    if(aspects!=""): aspects = aspects[3:]
    if(polarities!=""): polarities = polarities[3:]
        

    
    # Loại bỏ các câu null
    
    return aspects, polarities



# Create Streamlit app
st.title("Aspect-based sentiment analysis on IoT store comments")
st.write("This application was built based on the model we used to analyze data in our research. Although we still maintain the same framework as presented in the research process, some changes in the processing procedure have been made.")
st.write("Please input a short sample comment expressing your viewpoint on the experience at retail stores with IoT applications. Our model is built on English language data, so please enter an English comment to allow the model to classify it accurately.")
st.write("An example of a comment is: 'The smart cart is incredible'")
text = st.text_area("Text", height=200)

if st.button("Classify"):
    aspect, polarity = predict_multi_aspect(text)
    st.write("Sentiment: ", polarity)
    st.write("Aspect: ", aspect)
