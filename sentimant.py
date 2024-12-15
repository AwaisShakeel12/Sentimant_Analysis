import re
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors

# Custom dataset
data = [
    ("I love this product, it’s amazing!", "positive"),
    ("This is the best purchase I’ve ever made.", "positive"),
    ("Absolutely wonderful experience.", "positive"),
    ("Highly recommend this item to everyone!", "positive"),
    ("It works perfectly as described, very happy.", "positive"),
    ("Excellent quality and great customer service.", "positive"),
    ("This product exceeded my expectations.", "positive"),
    ("Very satisfied with the results, five stars.", "positive"),
    ("Fantastic performance, worth every penny.", "positive"),
    ("A great buy, I will definitely purchase again.", "positive"),
    ("I hate this product, it’s terrible.", "negative"),
    ("This is the worst purchase I’ve ever made.", "negative"),
    ("Completely disappointing experience.", "negative"),
    ("I would not recommend this item to anyone.", "negative"),
    ("It doesn’t work as advertised, very unhappy.", "negative"),
    ("Poor quality and bad customer service.", "negative"),
    ("This product failed to meet my expectations.", "negative"),
    ("Very dissatisfied with the results, one star.", "negative"),
    ("Terrible performance, a waste of money.", "negative"),
    ("A regretful buy, I will never purchase again.", "negative"),
]



# Preprocess text: clean and tokenize
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation, convert to lowercase
    return word_tokenize(text)


# Apply preprocessing
data = [(preprocess_text(text), label) for text, label in data]


word2vec = KeyedVectors.load_word2vec_format(r'E:\deep learning\NLP\nlp_start\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin', binary=True)



def review_to_vector(review):
    vectors = [word2vec[word] for word in review if word in word2vec]
    if vectors:
        return np.mean(vectors, axis=0)  
    else:
        return np.zeros(300)  

# Convert dataset
X = np.array([review_to_vector(text) for text, _ in data])  
y = np.array([1 if label == "positive" else 0 for _, label in data])  # Labels 1 = positive, 0 = negative




# train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict on test set
y_pred = classifier.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))


new_reviews = [
    "I absolutely love this product!",
    "Terrible experience, I hate it.",
    "Worth every penny, highly satisfied.",
    "Not what I expected, very disappointing.",
]




# Preprocess and convert to vectors
new_reviews_vectors = np.array([review_to_vector(preprocess_text(review)) for review in new_reviews])

# Predict sentiment
predictions = classifier.predict(new_reviews_vectors)

# Display results
for review, sentiment in zip(new_reviews, predictions):
    sentiment_label = "positive" if sentiment == 1 else "negative"
    print(f"Review: {review}\nPredicted Sentiment: {sentiment_label}\n")
