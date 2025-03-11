import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
#from textblob import TextBlob
import matplotlib.pyplot as plt
#from symspellpy import SymSpell, Verbosity
#import pkg_resources
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV ,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

#reproducibility
random.seed(42)
np.random.seed(42)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))  
lemmatizer = WordNetLemmatizer()

#accuracy at 0.76 does not improve plus it takes so much time 
'''
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
'''

'''
data = {
    "text" : [
        "I luv this movie!",
        "Terible film",
        "The film was awesome"
    ],
    "label": [1,0,1]
}

df = pd.DataFrame(data)
'''

#load datasets
df = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/train_dataset.csv")
df.rename(columns={"Text": "text", "Label": "label"}, inplace=True)

df_test = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/test_dataset.csv")
df_test.rename(columns={"Text": "text"}, inplace=True)

df_val = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/val_dataset.csv")
df_val.rename(columns={"Text": "text", "Label": "label"}, inplace=True)


#exploratory data analysis
print(df.describe())
sns.countplot(x='label', data=df)
plt.title("Class Distribution")
plt.show()


positive_words = " ".join(df[df["label"] == 1]["text"]).split()
negative_words = " ".join(df[df["label"] == 0]["text"]).split()

#count frequency 
pos_counts = Counter(positive_words)
neg_counts = Counter(negative_words)

pos_common = pos_counts.most_common(20)
neg_common = neg_counts.most_common(20)

pos_df = pd.DataFrame(pos_common, columns=["Word", "Count"])
neg_df = pd.DataFrame(neg_common, columns=["Word", "Count"])

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

#positive
axes[0].barh(pos_df["Word"], pos_df["Count"], color="green")
axes[0].invert_yaxis()  
axes[0].set_title("20 positive words")
axes[0].set_xlabel("Frequency")

#negative
axes[1].barh(neg_df["Word"], neg_df["Count"], color="red")
axes[1].invert_yaxis()
axes[1].set_title("20 negative words")
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.show()

avg_pos_words = df[df["label"] == 1]["text"].apply(lambda x: len(x.split())).mean()
avg_neg_words = df[df["label"] == 0]["text"].apply(lambda x: len(x.split())).mean()

plt.figure(figsize=(10, 4))
plt.barh(['Positive', 'Negative'], [avg_pos_words, avg_neg_words], height=0.5, color=['blue', 'red'])
plt.xticks(np.arange(0, max(avg_pos_words, avg_neg_words) + 10, 10))  
plt.xlabel('Average Number of Words')
plt.ylabel('Sentiment')
plt.title('Average Word Count in Positive and Negative Reviews')
plt.show()


#takes too much time 
'''
def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())
'''
'''
def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text
'''

#found most common "slang" words and mistakes
corrections = {
    "4all": "for all",
    "u": "you",
    "r": "are",
    "ur": "your",
    "b4": "before",
    "gr8": "great",
    "thx": "thanks",
    "pls": "please",
    "idk": "I don't know",
    "gonna": "going to",
    "wanna": "want to",
    "cuz": "because",
    "cos": "because",
    "lemme": "let me",
    "gimme": "give me",
    "aint": "is not",
    "dunno": "do not know",
    "lol": "", 
    "teh": "the",
    "hte": "the",
    "jsut": "just",
    "dont": "don't",
    "doesnt": "doesn't",
    "cant": "can't",
    "wont": "won't",
    "havent": "haven't",
    "im": "I'm",
    "ive": "I've",
    "its": "it's",
    "alot": "a lot",
    "thier": "their",
    "adress": "address",
    "occurence": "occurrence",
    "definately": "definitely",
    "seperate": "separate",
    "recieve": "receive",
    "wierd": "weird",
    "untill": "until",
    "loose": "lose",
    "truely": "truly",
    "your": "you're",
    "their": "they're",
    "there": "their",
    "then": "than",
    "could of": "could have",
    "should of": "should have",
    "would of": "would have"
}

def correct_text(text):
    if not isinstance(text, str):
        return text  
    
    words = text.split()  
    corrected_words = [corrections.get(word.lower(), word) for word in words]
    
    return " ".join(corrected_words)




def lemmatize_text(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]


re_negation = re.compile(r"n't\b")

def negation_abbreviated_to_standard(sent):
    return re_negation.sub(" not", sent)

def preprocess_text(text):
    if isinstance(text, str):
        text = negation_abbreviated_to_standard(text)

        #lowercasing
        text = text.lower()
        
        #removing special characters,numbers and extra spacing
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        patern = r'[^a-zA-Z\s]'
        text = re.sub(patern,'',text)

        #tokenization
        #tokens = text.split()  
        tokens = word_tokenize(text)

        #stopwords removal
        tokens = [word for word in tokens if word not in stop_words]

        #lemmatization
        tokens = lemmatize_text(tokens)

        return " ".join(tokens)
    return string



#apply preprocessing to the 3 datasets
df["text"] = df["text"].apply(lambda x: preprocess_text(correct_text(x)))
df_test["text"] = df_test["text"].apply(lambda x: preprocess_text(correct_text(x)))
df_val["text"] = df_val["text"].apply(lambda x: preprocess_text(correct_text(x)))


#splitting the data set
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

#TF-IDF Method 
#preprocessed all the words in order to improve the method 
#converting to lowercase, removing stopwords, removing special characters can improve the process
vectorizer = TfidfVectorizer(
    max_df=0.7,  #ignore the words that appear 80% in the texts 
    min_df=10,  #ignore the words that appear at most 5 times in the texts
    ngram_range=(1,2),  #bigram
    stop_words="english"
    )
X_train_tfidf = vectorizer.fit_transform(X_train)  
X_test_tfidf = vectorizer.transform(X_test)
X_val_tfidf = vectorizer.transform(df_val["text"])


#feature scaling
scaler = StandardScaler(with_mean=False)
X_train_tfidf_scaled = scaler.fit_transform(X_train_tfidf)
X_test_tfidf_scaled = scaler.transform(X_test_tfidf)
X_val_tfidf_scaled = scaler.transform(X_val_tfidf)


#model = LogisticRegression(max_iter=2000, solver='saga', C=1.0, random_state=42)
#log_reg = LogisticRegression(solver='saga', random_state=42)
'''
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [500, 1000]
}
'''
#hyperparameter with GridSearchCV and Logistic Regreation Model 
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'saga'], 'max_iter': [2000, 5000]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, scoring='accuracy')
model.fit(X_train_tfidf_scaled, y_train)

'''
model = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
model.fit(X_train_tfidf, y_train)
'''

print(f"Best Parameters: {model.best_params_}")


#predictions
y_pred = model.predict(X_test_tfidf_scaled)

#evaluation 
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy: .2f}")
print("Classification Report: \n ", classification_report(y_test,y_pred))

#cross validation accuracy
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring="accuracy")
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")




#test data set



#df_test["text"] = df_test["text"].apply(preprocess_text)
#X_test_tfidf = vectorizer.transform(df_test["text"])
#y_test_pred = model.predict(X_test_tfidf)
#df_test["predicted_label"] = y_test_pred
df_test["predicted_label"] = model.predict(vectorizer.transform(df_test["text"]))
df_test_output = df_test[["ID", "predicted_label"]]
df_test_output.to_csv("/home/erginadimitraina/AI2/test_results.csv", index=False)
print(df_test.head())  
print(df_test.columns)  

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

#ROC Curve
y_scores = model.decision_function(X_test_tfidf_scaled)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#generate learning curve
train_sizes, train_scores, test_scores = learning_curve(LogisticRegression(), X_train_tfidf, y_train, cv=kfold, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10))
train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_mean, label="Training Score", color="blue", marker="o")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.plot(train_sizes, test_mean, label="Validation Score", color="red", marker="o")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve For Logistic Regression")
plt.legend()
plt.show()

