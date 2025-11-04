import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


#  Load Data
df = pd.read_csv("dataset/StudentsPerformance.csv")

# Keep what I want
df = df[["gender", "parental level of education", "test preparation course", "math score"]]


# Encode text features
from sklearn.preprocessing import OneHotEncoder


X = df[["gender", "parental level of education", "test preparation course"]]
y = df["math score"]

# One-hot encode
encoder = OneHotEncoder(drop="first", sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Train a simple linear regression model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("R² score on test:", model.score(X_test, y_test))


# Bundle and save model
bundle = {"model": model, "encoder": encoder}
with open("student_score_model.pkl", "wb") as f:
    pickle.dump(bundle, f)