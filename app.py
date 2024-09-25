from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

# Preprocessing function to clean and process text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation and non-alphabetical characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the tokens (reduce words to their root forms)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a string
    return ' '.join(tokens)

# Function to get response based on user input and similarity
def get_response(user_prompt, data, vectorizer, temperature=0.5, similarity_threshold=0.3):
    # Preprocess the user input
    user_prompt = preprocess_text(user_prompt)

    # Vectorize the dataset prompts and user input
    X = vectorizer.transform(data['prompt'])
    user_vec = vectorizer.transform([user_prompt])

    # Compute cosine similarity between user input and dataset prompts
    similarities = cosine_similarity(user_vec, X)

    # Get the index of the most similar prompt
    most_similar_index = similarities.argmax()

    # Check if the highest similarity score is above the threshold
    if similarities[0, most_similar_index] < similarity_threshold:
        return "I'm sorry, I don't have an answer for that right now."

    # Get the corresponding response
    response = data.iloc[most_similar_index]['response']

    # Modify the response slightly based on the temperature (adds randomness)
    if temperature > 0.5:
        words = response.split()
        random.shuffle(words)
        response = ' '.join(words)

    return response

# Load the dataset and initialize the vectorizer
file_path = 'expanded_faq_data.csv'
data = pd.read_csv(file_path)

# Preprocess the 'prompt' column in the dataset
data['prompt'] = data['prompt'].apply(preprocess_text)

# Initialize the vectorizer and fit on the preprocessed prompts
vectorizer = TfidfVectorizer()
vectorizer.fit(data['prompt'])

# API endpoint to handle chatbot requests
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get user input from the request body (JSON)
        user_input = request.json.get('prompt')

        if not user_input:
            return jsonify({"error": "No prompt provided"}), 400

        # Get the chatbot response
        response = get_response(user_input, data, vectorizer)

        # Return the response as JSON
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
