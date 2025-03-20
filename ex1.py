import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import random
from nltk.corpus import stopwords,opinion_lexicon
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss
import collections
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD


#reproducibility
#this is used, if someone else runs the model it will produce the same results
random.seed(42)
np.random.seed(42)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('opinion_lexicon')

nltk_stop_words = set(stopwords.words('english'))  
#lemmatizer = WordNetLemmatizer()

#accuracy at 0.76 does not improve plus it takes so much time 
'''
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
'''


#load datasets
df = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/train_dataset.csv")
df.rename(columns={"Text": "text", "Label": "label"}, inplace=True)

df_test = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/test_dataset.csv")
df_test.rename(columns={"Text": "text"}, inplace=True)

df_val = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/val_dataset.csv")
df_val.rename(columns={"Text": "text", "Label": "label"}, inplace=True)


#exploratory data analysis
#plot some data for better understanding
print(df.describe())
sns.countplot(x='label', data=df)
plt.title("Class Distribution")
plt.show()

#before preprocess so the outcome we expect it will be mostly stopwords in both sentiments, which indicates us that stopwords really influence the model 
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
axes[0].set_title("20 positive words before the preprocessing")
axes[0].set_xlabel("Frequency")

#negative
axes[1].barh(neg_df["Word"], neg_df["Count"], color="red")
axes[1].invert_yaxis()
axes[1].set_title("20 negative words before the preprocessing")
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
    "im": "",
    "today": "",
    "one": "",
    "got": "",
    "going": "",
    "amp": "",
    "youre":"you are"
    }

def correct_text(text):
    if not isinstance(text, str):
        return text  
    
    words = text.split()  
    corrected_words = [corrections.get(word.lower(), word) for word in words]
    
    return " ".join(corrected_words)


custom_stopwords = {
    "to","I","the","a","my","i","and","is","in","for","of","it","on","have","you","so","me","but","that","not","you","with","be","im","now","IM","amp","up","go","get","this","with","just","I'm","was","at","be","out","all","are","work","now","got","do","day","back"
}
#combined_stop_words = custom_stopwords | nltk_stop_words

#def lemmatize_text(tokens):
    #return [lemmatizer.lemmatize(word) for word in tokens]

'''
re_negation = re.compile(r"n't\b")

def negation_abbreviated_to_standard(sent):
    return re_negation.sub(" not", sent)


'''


def preprocess_text(text):
    if not isinstance(text, str):  
        return ""  
    
    text = correct_text(text)
    text = text.lower()  
    text = re.sub(r"http\S+", "", text) #urls
    text = re.sub(r"\d+", "", text)  #numbers
    text = re.sub(r"[^\w\s]", "", text)  #simeia stiksis
    text = re.sub(r"\s+", " ", text).strip()  #kena
    
    #tokens = word_tokenize(text)  
    #tokens = [word for word in tokens if word not in stop_words]  
    #tokens = [word for word in tokens if len(word) > 2]  
    #tokens = [lemmatizer.lemmatize(word) for word in tokens] 
    tokens = text.split()  
    tokens = [word for word in tokens if word not in custom_stopwords] 
    
    return " ".join(tokens)

'''
df["text"] = df["text"].apply(lambda x: preprocess_text(x))
df_test["text"] = df_test["text"].apply(lambda x: preprocess_text(x))
df_val["text"] = df_val["text"].apply(lambda x: preprocess_text(x))
'''
df["text"] = df["text"].apply(preprocess_text)
df_test["text"] = df_test["text"].apply(preprocess_text)
df_val["text"] = df_val["text"].apply(preprocess_text)





#after the preprocess I plot the most common positive and negative words 

all_words = " ".join(df["text"]).split()
word_counts = Counter(all_words) 
positive_lexicon = set(opinion_lexicon.positive())  
negative_lexicon = set(opinion_lexicon.negative())  

positive_words = {word: count for word, count in word_counts.items() if word in positive_lexicon}
negative_words = {word: count for word, count in word_counts.items() if word in negative_lexicon}

pos_common = Counter(positive_words).most_common(20)
neg_common = Counter(negative_words).most_common(20)

pos_df = pd.DataFrame(pos_common, columns=["Word", "Count"])
neg_df = pd.DataFrame(neg_common, columns=["Word", "Count"])

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].barh(pos_df["Word"], pos_df["Count"], color="green")
axes[0].invert_yaxis()  
axes[0].set_title("20 Most Common Positive Words After Preprocessing")
axes[0].set_xlabel("Frequency")

axes[1].barh(neg_df["Word"], neg_df["Count"], color="red")
axes[1].invert_yaxis()
axes[1].set_title("20 Most Common Negative Words After Preprocessing")
axes[1].set_xlabel("Frequency")

plt.tight_layout()
plt.show()


#word cloud 
pos_text = " ".join(df[df["label"] == 1]["text"])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Positive Reviews")
plt.show()


neg_text = " ".join(df[df["label"] == 0]["text"])
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(neg_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Negative Reviews")
plt.show()

'''
#pattern detection
vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english')
X = vectorizer.fit_transform(df["text"])
sum_words = X.sum(axis=0).A1
words_freq = sorted([(word, sum_words[idx]) for word, idx in vectorizer.vocabulary_.items()], key=lambda x: x[1], reverse=True)[:20]
freq_df = pd.DataFrame(words_freq, columns=['Bigram', 'Count'])
plt.figure(figsize=(12, 5))
plt.barh(freq_df["Bigram"], freq_df["Count"], color='orange')
plt.xlabel("Frequency")
plt.ylabel("Bigrams")
plt.title("Top 20 Most Frequent Bigrams After Preprocessing")
plt.gca().invert_yaxis()
plt.show()
'''

#splitting the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
#splitting the training data set to validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)


#TF-IDF Method 
vectorizer = TfidfVectorizer(max_df=0.7, min_df=10, ngram_range=(1,2), stop_words=list(custom_stopwords))
X_train_tfidf = vectorizer.fit_transform(X_train)  
X_test_tfidf = vectorizer.transform(X_test)
X_val_tfidf = vectorizer.transform(X_val)



'''
#feature scaling
scaler = StandardScaler(with_mean=False)
X_train_tfidf_scaled = scaler.fit_transform(X_train_tfidf)
X_test_tfidf_scaled = scaler.transform(X_test_tfidf)
X_val_tfidf_scaled = scaler.transform(X_val_tfidf)
'''

#hyperparameter with GridSearchCV and Logistic Regreation Model 
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'saga'], 'max_iter': [3000, 5000]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, scoring='accuracy')
model.fit(X_train_tfidf, y_train)


print(f"Best Parameters: {model.best_params_}")


#predictions
y_pred = model.predict(X_train_tfidf)

#evaluation 
accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy: .2f}")
print("Classification Report: \n ", classification_report(y_test,y_pred))


#cross validation accuracy for testing purposes
cv_scores = cross_val_score(model.best_estimator_, X_train_tfidf, y_train, cv=kfold, scoring="accuracy")
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

'''
#this is added beacause the word cloud especially on the negative sentiment plots words that do not give information about the sentiment 
#and they are very neutral e.x. today, tommorw , going etc 
#thats why i used this part to keep the words that only have a high IDF number in the word cloud and not focus on the ones that do not give a clear sentiment.
feature_array = np.array(vectorizer.get_feature_names_out())
idf_values = vectorizer.idf_
#sort words based on the higher IDF
sorted_indices = np.argsort(idf_values)[::-1] 
top_n = 100  
important_words = feature_array[sorted_indices][:top_n]
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(important_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
'''


X_test_final_tfidf = vectorizer.transform(df_test["text"])
#X_test_final_tfidf_scaled = scaler.transform(X_test_final_tfidf)
df_test["predicted_label"] = model.predict(X_test_final_tfidf)
df_test_output = df_test[["ID", "predicted_label"]]
df_test_output.to_csv("/home/erginadimitraina/AI2/test_results.csv", index=False)
 

'''
#loss function for overfitting and underfitting
y_train_probs = model.predict_proba(X_train_tfidf_scaled)[:, 1]  
y_val_probs = model.predict_proba(X_val_tfidf_scaled)[:, 1]


train_loss = log_loss(y_train, y_train_probs)
val_loss = log_loss(y_val, y_val_probs)

print(f"Train Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
'''

'''
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
train_sizes, train_scores, test_scores = learning_curve(
    model.best_estimator_, 
    X_train_tfidf_scaled, 
    y_train, 
    cv=kfold, 
    scoring="accuracy", 
    train_sizes=np.linspace(0.1, 1.0, 20) 
)

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


train_losses = []
val_losses = []
train_sizes = np.linspace(0.1, 1.0, 10)

best_model = model.best_estimator_  

for size in train_sizes:
    subset_size = int(size * X_train_tfidf_scaled.shape[0])  # Χρησιμοποιούμε shape[0] αντί για len()
    
    X_subset, y_subset = X_train_tfidf_scaled[:subset_size], y_train[:subset_size]

    best_model.fit(X_subset, y_subset)
    
    y_subset_probs = best_model.predict_proba(X_subset)[:, 1]
    y_val_probs = best_model.predict_proba(X_val_tfidf_scaled)[:, 1]
    
    train_losses.append(log_loss(y_subset, y_subset_probs))
    val_losses.append(log_loss(y_val, y_val_probs))

plt.plot(train_sizes, train_losses, label="Train Loss", marker="o", color="blue")
plt.plot(train_sizes, val_losses, label="Validation Loss", marker="o", color="red")
plt.xlabel("Training Data Size")
plt.ylabel("Log Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()'
'''