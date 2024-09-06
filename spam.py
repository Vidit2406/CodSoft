import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_curve
import numpy as np
import joblib

# Read the CSV file
data = pd.read_csv(r"C:\Coding\Python\cleaned_spam.csv")

# Drop any rows with NaN values in 'message'
data = data.dropna(subset=['message'])

# Features and target variable
X = data['message']
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression(C=100, penalty='l2', max_iter=1000))
])

# Train the model
print("Training the model...")
pipeline.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(pipeline, 'spam_classifier_model.pkl')

# Predict probabilities
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Find a balance threshold manually, e.g., by using the maximum f1-score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_index = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_index]
print(f"\nBest threshold based on F1-score: {best_threshold}")

# Set predictions based on the best threshold
y_pred = (y_proba >= best_threshold).astype(int)

# Evaluate the model with the new threshold
print("\nClassification Report with adjusted threshold:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
