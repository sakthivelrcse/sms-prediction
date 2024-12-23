from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

# Example data (replace with your actual data)
X = [
    "Free money now!",
    "Hey, how are you doing?",
    "Limited time offer, act fast!",
    "Let's meet for lunch tomorrow",
    "Exclusive offer just for you"
]
y = [1, 0, 1, 0, 1]  # Labels for spam (1) and not spam (0)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the TfidfVectorizer
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the trained vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer have been trained and saved.")
