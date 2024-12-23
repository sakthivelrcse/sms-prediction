# In your Streamlit app (e.g., app.py)

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.utils.validation import check_is_fitted, NotFittedError  # Import NotFittedError

ps = PorterStemmer()

# Function to preprocess and clean the text
def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric words
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stem the words using Porter Stemmer
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))  # Load the saved vectorizer
    model = pickle.load(open('model.pkl', 'rb'))  # Load the trained model
    
    # Check if the vectorizer is fitted
    check_is_fitted(tfidf)
    
except (FileNotFoundError, NotFittedError) as e:
    st.error("The vectorizer or model file is not found or not properly fitted. Please ensure the model and vectorizer are correctly saved.")
    raise e

# Streamlit UI
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)
    
    try:
        # Vectorize the input using the loaded vectorizer
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict the result using the model
        result = model.predict(vector_input)[0]
        
        # Display the result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    
    except NotFittedError:
        st.error("The vectorizer is not fitted. Please train the vectorizer and try again.")
