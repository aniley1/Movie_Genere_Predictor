import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
data = pd.read_csv('Indian_movies.csv')  # Replace 'Indian_movies.csv' with the actual file path if it's located elsewhere

# Split the data into training and testing sets
X = data['Description']
y = data['Genere']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train the SVM Classifier
svm_classifier = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

# Make Predictions
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Classification Report:\n{report}')

# Predict Genres for New Movie Plot Summaries
new_plot_summaries = ["A group of friends go on an adventure", "A sci-fi story about time travel"]
new_tfidf = tfidf_vectorizer.transform(new_plot_summaries)
predicted_genres = svm_classifier.predict(new_tfidf)

print("Predicted Genres for New Movie Plot Summaries:")
for Description, Genere in zip(new_plot_summaries, predicted_genres):
    print(f"Plot Summary: {Description} -> Predicted Genre: {Genere}")
