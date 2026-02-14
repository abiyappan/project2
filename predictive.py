# =====================================
# TITANIC SURVIVAL PREDICTION PROJECT
# WITH VISUAL OUTPUT
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Program Started...\n")

# Load Dataset
df = pd.read_csv("titanic.csv")
print("Dataset Loaded Successfully!\n")

# Drop unnecessary columns
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle Missing Values properly
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert categorical to numeric
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

print("Missing Values After Cleaning:\n", df.isnull().sum(), "\n")

# =============================
# 🔥 VISUALIZATION SECTION
# =============================

# 1️⃣ Survival Count Plot
plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# 2️⃣ Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# =============================
# MACHINE LEARNING PART
# =============================

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model Training Completed!\n")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy * 100, 2), "%\n")

print("Classification Report:\n", classification_report(y_test, y_pred))

# 3️⃣ Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Project Completed Successfully!")

