import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('xgb_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to make a prediction
def predict_churn(user_input):
    # Convert the input to a DataFrame
    user_df = pd.DataFrame([user_input])

    # Encode categorical variables (like we did in training)
    user_df = pd.get_dummies(user_df, columns=['Geography', 'Gender'], drop_first=True)

    # Reindex the DataFrame so it matches the model's training data format
    user_df = user_df.reindex(columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                                       'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                                       'Geography_Germany', 'Geography_Spain', 'Gender_Male'], 
                              fill_value=0)

    # Scale the input features
    user_df_scaled = scaler.transform(user_df)

    # Make the prediction (Random Forest)
    prediction = model.predict(user_df_scaled)

    return "Churn" if prediction[0] == 1 else "No Churn"

# Example input from a user (this can be changed)
if __name__ == "__main__":
    user_input = {
        'CreditScore': 600,
        'Age': 40,
        'Tenure': 3,
        'Balance': 60000,
        'NumOfProducts': 1,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 50000,
        'Geography': 'Germany',  # Must match training categories
        'Gender': 'Male'
    }

    result = predict_churn(user_input)
    print("Prediction for user input:", result)
