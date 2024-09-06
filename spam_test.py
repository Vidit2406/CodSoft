import joblib
import pandas as pd

# Load the saved model
pipeline = joblib.load('spam_classifier_model.pkl')

# Define a function to predict if a message is spam
def predict_message(message):
    # Predict probability
    proba = pipeline.predict_proba([message])[0, 1]
    
    # Load the best threshold used during training
    best_threshold = 0.2624108452843914  # Replace with the threshold you calculated during training

    # Predict based on the best threshold
    prediction = (proba >= best_threshold).astype(int)
    
    return 'spam' if prediction == 1 else 'ham'

# Test the function with a new message
new_message = "D/PYouarerequestedtodeposityourwardfee Expert Bull Market Analysis Link: chat.whatsapp.com/Caj8cxI1TfUARFM3hD40P5 -KCMWSP"
result = predict_message(new_message)
print(f"The message is: {result}")
