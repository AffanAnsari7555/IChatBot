# app.py
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
