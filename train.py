import pandas as pd

# Load data
data = pd.read_csv("data/data.csv")

# Verify dataset structure
print(data.head())
print(data.info())

# Split features and target
X = data[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]  # Explicitly specify features
y = data["Species"]

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
model_path = 'models/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print("Model has been trained successfully.")