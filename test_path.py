import os

file_path = r'C:\Users\yashw\OneDrive\Desktop\BrainStrokePrediction\backend\stroke_data.csv'

if os.path.exists(file_path):
    print("File exists and is accessible:", file_path)
else:
    print("File not found or inaccessible:", file_path)
