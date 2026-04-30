import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

file_path = r"D:\sneha\smartshield_new\processed_html_dataset.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("Total rows before cleaning:", len(df))

# 🚨 REMOVE NaN rows
df = df.dropna(subset=["clean_html"])

# 🚨 REMOVE empty strings
df = df[df["clean_html"].str.strip() != ""]

print("Total rows after removing empty text:", len(df))

X = df["clean_html"]
y = df["status"]

print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_vec = vectorizer.fit_transform(X)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print("✅ Model Accuracy:", score)

# Save model
model_path = r"D:\sneha\smartshield_new\models\html_model.pkl"
vectorizer_path = r"D:\sneha\smartshield_new\models\html_vectorizer.pkl"

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("✅ Model saved successfully")