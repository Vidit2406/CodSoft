import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import xgboost as xgb  # Import XGBoost library

# Load the dataset
df = pd.read_csv(r'D:\Dataset\Churn\Churn_Modelling.csv')

# Drop columns that are not useful for prediction
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Convert categorical variables into numerical ones using one-hot encoding
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Define features (X) and target variable (y)
X = df.drop(columns=['Exited'])
y = df['Exited']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler to standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit to training data and transform
X_test = scaler.transform(X_test)        # Transform test data

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Initialize the XGBoost Classifier with basic hyperparameters
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and scaler for future use
joblib.dump(model, 'xgb_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler have been saved successfully.")
