import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import pad_sequences
import streamlit as st

word_index = imdb.get_word_index()
mapping_index_to_words = { v:k for (k,v) in word_index.items() }

from tensorflow.keras.models import load_model
model = load_model('simple_rnn_imdb.h5')

def decode(encoded_review):
    return ' '.join([mapping_index_to_words.get(i-3,"?") for i in encoded_review]) 

def preprocess(text):
    words = text.lower().split()
    encoded_review = [(word_index.get(word,2) + 3) for word in words]
    padded_review = pad_sequences([encoded_review], maxlen = 500)
    return padded_review

def predict_sentiment(review):
    encoded_review = preprocess(review)
    prediction = model.predict([encoded_review])
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment,prediction[0][0]


st.title("Movie Review Sentiment Analysis")
st.write("Enter a review on a movie to predict its sentiment (positive / negative)")
user_inp = st.text_area("Movie Review")
if st.button("Classify"):
    sentiment, prob = predict_sentiment(user_inp)
    if sentiment and prob:
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Sentiment score (0=negative to 1=positive) : {prob}")
else:
    st.write("")
