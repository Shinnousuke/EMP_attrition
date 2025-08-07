# train_model.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# --- Step 1: Load Dataset ---
df = pd.read_csv("C:/Users/Admin/Desktop/internship_projects/attention_detection_employees/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# --- Step 2: Encode Categorical Variables ---
le = LabelEncoder()
for column in df.select_dtypes(include='object').columns:
    df[column] = le.fit_transform(df[column])

# --- Step 3: Split Features and Target ---
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# --- Step 4: Split Dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 5: Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 6: Evaluate ---
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Step 7: Save Model ---
model_path = "C:/Users/Admin/Desktop/internship_projects/attention_detection_employees/rf_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# --- Step 8: Save Trained Columns (feature names) ---
columns_path = "C:/Users/Admin/Desktop/internship_projects/attention_detection_employees/trained_columns.pkl"
with open(columns_path, "wb") as f:
    pickle.dump(list(X.columns), f)

print(f"✅ Model saved to {model_path}")
print(f"✅ Trained columns saved to {columns_path}")
