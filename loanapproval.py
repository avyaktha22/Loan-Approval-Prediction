import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
# Replace 'loan_data.csv' with your dataset file path
df = pd.read_csv('loan_data.csv')

# Preview the data
print(df.head())

# Data Preprocessing
# Fill missing values or drop them if necessary
df.fillna(method='ffill', inplace=True)

# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split the data into features (X) and target (y)
X = df.drop('Loan_Status', axis=1)  # Replace 'Loan_Status' with your target column
y = df['Loan_Status']  # Replace 'Loan_Status' with your target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
