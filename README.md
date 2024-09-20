# IChatBot

import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## Preload the dataset from the given file path
def load_data(file_path):
    return pd.read_csv(file_path)

## Function to get response based on user input and similarity
def get_response(user_prompt, data, vectorizer, temperature=0.5):
    # Vectorize the dataset prompts and user input
    X = vectorizer.transform(data['prompt'])
    user_vec = vectorizer.transform([user_prompt])
    
    # Compute cosine similarity between user input and dataset prompts
    similarities = cosine_similarity(user_vec, X)
    
    # Find the index of the most similar prompt
    most_similar_index = similarities.argmax()
    
    # Get the corresponding response
    response = data.iloc[most_similar_index]['response']
    
    # Modify the response slightly based on the temperature (adds some randomness)
    if temperature > 0.5:
        # Introduce a bit of randomness by shuffling the sentence slightly
        words = response.split()
        random.shuffle(words)
        response = ' '.join(words)
    
    return response

## Streamlit App Configuration
st.set_page_config(page_title="Institute Chatbot", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')

st.header("Institute Chatbot 🤖")

## Specify the file path of your dataset
file_path = 'C:/chatbot/expanded_faq_data.csv'  # Make sure to correct this file path

## Preload the dataset containing institute prompts and responses
data = load_data(file_path)

## Initialize the vectorizer
vectorizer = TfidfVectorizer()

## Fit the vectorizer on the dataset prompts
vectorizer.fit(data['prompt'])

user_prompt = st.text_input("Ask a question about the institute")

if st.button("Get Response"):
    response = get_response(user_prompt, data, vectorizer)
    st.write(response)
