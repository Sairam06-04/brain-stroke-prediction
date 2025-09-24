import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset (replace with the actual path to your dataset)
data = pd.read_csv('C:\\Users\\yashw\\OneDrive\\Desktop\\BrainStrokePrediction\\backend\\stroke_data.csv')


# Preprocessing: Handle missing values and encode categorical variables
# Fill missing BMI values with the mean
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())
  # Fill missing BMI values with the mean
data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)  # Encode gender (Male=1, Female=0)
data['smoking_status'] = data['smoking_status'].map({
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2
})

# Select features and target variable
X = data[['age', 'gender', 'bmi', 'smoking_status']]  # Features
y = data['stroke']  # Target (1: Stroke, 0: No stroke)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
with open('stroke_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as 'stroke_model.pkl'")
