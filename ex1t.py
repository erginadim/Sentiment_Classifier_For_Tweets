import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# ---- Αρχικοποίηση seeds για αναπαραγωγή των αποτελεσμάτων ----
random.seed(42)
np.random.seed(42)

# ---- Κατέβασμα των απαραίτητων nltk πακέτων ----
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('opinion_lexicon')

# ---- Φόρτωση των δεδομένων ----
df = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/train_dataset.csv")
df.rename(columns={"Text": "text", "Label": "label"}, inplace=True)

df_test = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/test_dataset.csv")
df_test.rename(columns={"Text": "text"}, inplace=True)

df_val = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/val_dataset.csv")
df_val.rename(columns={"Text": "text", "Label": "label"}, inplace=True)


stop_words = set(stopwords.words('english'))  

# ---- Συνάρτηση προεπεξεργασίας κειμένου ----
def preprocess_text(text):
    if not isinstance(text, str):  
        return ""  
    text = text.lower()  
    text = re.sub(r"http\S+", "", text)  # Αφαίρεση URL
    text = re.sub(r"\d+", "", text)  # Αφαίρεση αριθμών
    text = re.sub(r"[^\w\s]", "", text)  # Αφαίρεση σημείων στίξης
    text = re.sub(r"\s+", " ", text).strip()  # Αφαίρεση περιττών κενών
    tokens = word_tokenize(text)  
    tokens = [word for word in tokens if word not in stop_words]  
    return " ".join(tokens)

# ---- Εφαρμογή της προεπεξεργασίας ----
df["text"] = df["text"].apply(preprocess_text)
df_test["text"] = df_test["text"].apply(preprocess_text)
df_val["text"] = df_val["text"].apply(preprocess_text)

# ---- Διαχωρισμός σε Training & Validation ----
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# ---- TF-IDF Vectorization ----
vectorizer = TfidfVectorizer(max_df=0.7, min_df=10, ngram_range=(1,2), stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)  
X_test_tfidf = vectorizer.transform(X_test)
X_val_tfidf = vectorizer.transform(X_val)

# ---- Feature Scaling (Προαιρετικό) ----
scaler = StandardScaler(with_mean=False)
X_train_tfidf_scaled = scaler.fit_transform(X_train_tfidf)
X_test_tfidf_scaled = scaler.transform(X_test_tfidf)
X_val_tfidf_scaled = scaler.transform(X_val_tfidf)

# ---- Grid Search για Hyperparameter Tuning ----
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'saga'], 'max_iter': [2000, 5000]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, scoring='accuracy')
model.fit(X_train_tfidf_scaled, y_train)

# ---- Βέλτιστες Παράμετροι ----
print(f"Best Parameters: {model.best_params_}")

# ---- Αξιολόγηση στο Test Set ----
y_pred = model.predict(X_test_tfidf_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---- Προβλέψεις στο Test Dataset ----
X_test_final_tfidf = vectorizer.transform(df_test["text"])
X_test_final_tfidf_scaled = scaler.transform(X_test_final_tfidf)
df_test["predicted_label"] = model.predict(X_test_final_tfidf_scaled)

# ---- Αποθήκευση αποτελεσμάτων ----
df_test_output = df_test[["ID", "predicted_label"]]
df_test_output.to_csv("/home/erginadimitraina/AI2/test_results.csv", index=False)