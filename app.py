import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to get response based on user input and similarity
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

# Streamlit App Configuration
st.set_page_config(page_title="Institute Chatbot", page_icon='🤖', layout='centered', initial_sidebar_state='collapsed')

st.header("Institute Chatbot 🤖")

# Allow the user to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    # Preload the dataset from the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully!")

    # Initialize the vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the dataset prompts
    vectorizer.fit(data['prompt'])

    user_prompt = st.text_input("Ask a question about the institute")

    if st.button("Get Response"):
        response = get_response(user_prompt, data, vectorizer)
        st.write(response)
else:
    st.write("Please upload a CSV file to proceed.")
