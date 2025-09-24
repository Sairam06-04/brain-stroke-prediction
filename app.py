import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

app = Flask(__name__)

# Create a folder to store static files
os.makedirs('static', exist_ok=True)

# Load the trained model
try:
    model = pickle.load(open('stroke_model.pkl', 'rb'))
except FileNotFoundError:
    raise Exception("Model file 'stroke_model.pkl' not found. Train the model and save it in the backend folder.")

@app.route('/batch_predict_visualize', methods=['POST'])
def batch_predict_visualize():
    # Get the uploaded CSV file
    file = request.files['file']
    data = pd.read_csv(file)

    # Preprocess the data (you can customize this based on your model)
    data['bmi'].fillna(data['bmi'].mean(), inplace=True)
    data['gender'] = data['gender'].apply(lambda x: 1 if x == 'Male' else 0)  # Encode gender

    # Make predictions
    predictions = model.predict(data)
    data['Stroke Risk'] = np.where(predictions == 1, 'Risk', 'No Risk')

    # Save predictions to a CSV file
    predictions_file = 'static/predictions.csv'
    data.to_csv(predictions_file, index=False)

    # Generate visualizations
    visualization_urls = {}

    # Visualization 1: Stroke Risk by Age Group
    plt.figure(figsize=(10, 6))
    sns.histplot(data[data['Stroke Risk'] == 'Risk']['age'], bins=10, kde=True, color='red', label='Stroke Risk')
    plt.title('Stroke Risk by Age Group')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()
    age_chart = 'static/stroke_by_age.png'
    plt.savefig(age_chart)
    plt.close()
    visualization_urls['age'] = age_chart

    # Visualization 2: Stroke Risk by Gender
    plt.figure(figsize=(8, 6))
    sns.countplot(x='gender', hue='Stroke Risk', data=data)
    plt.title('Stroke Risk by Gender')
    plt.xlabel('Gender (0 = Female, 1 = Male)')
    plt.ylabel('Count')
    gender_chart = 'static/stroke_by_gender.png'
    plt.savefig(gender_chart)
    plt.close()
    visualization_urls['gender'] = gender_chart

    # Visualization 3: Stroke Risk by BMI
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Stroke Risk', y='bmi', data=data, palette='coolwarm')
    plt.title('Stroke Risk by BMI')
    plt.xlabel('Stroke Risk')
    plt.ylabel('BMI')
    bmi_chart = 'static/stroke_by_bmi.png'
    plt.savefig(bmi_chart)
    plt.close()
    visualization_urls['bmi'] = bmi_chart

    # Visualization 4: Stroke Risk by Smoking Status
    plt.figure(figsize=(8, 6))
    sns.countplot(x='smoking_status', hue='Stroke Risk', data=data, palette='Set2')
    plt.title('Stroke Risk by Smoking Status')
    plt.xlabel('Smoking Status')
    plt.ylabel('Count')
    smoking_chart = 'static/stroke_by_smoking.png'
    plt.savefig(smoking_chart)
    plt.close()
    visualization_urls['smoking_status'] = smoking_chart

    # Return response
    return jsonify({
        'predictions_url': predictions_file,
        'visualization_urls': visualization_urls
    })
    

if __name__ == '__main__':
    app.run(debug=True)
