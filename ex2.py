'''
EXPERIMENTS
-> hidden dim try: 64,100,130,160,200,300.
-> learning rates try:0.01, 0.001, 0.0005
-> epochs try:20,30,40,45,50,55 best one 50
-> Data augmentation
Optimization
Dropout layer: nn.Dropout(0.3) ανάμεσα στα layers
early stopping 
'''

#the programm is not steady 


import pandas as pd
import numpy as np
import random
import re
import nltk
import gensim
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score 
from gensim.models import Word2Vec
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix,roc_curve, auc
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import opinion_lexicon,words
from collections import Counter
from textblob import TextBlob

#reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#nltk.download('words')

stemmer = PorterStemmer()
#load datasets
df = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/train_dataset.csv")
df.rename(columns={"Text": "text", "Label": "label"}, inplace=True)

df_test = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/test_dataset.csv")
df_test.rename(columns={"Text": "text"}, inplace=True)

df_val = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/val_dataset.csv")
df_val.rename(columns={"Text": "text", "Label": "label"}, inplace=True)


#I follow the same preprocess as the project 1 but I tried to improve it based on the feedback we got 
english_words_set = set(words.words())
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
    "LOL": "",
    "im": "",
    "today": "",
    "one": "",
    "got": "",
    "going": "",
    "amp": "",
    "youre":"you are",
    "haha": "laugh",
    "ha": "laugh" ,
    #"quot": ""
    }

'''
def is_english_word(word):
    return word in english_words_set

def detect_language(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except:
        return False
'''

def correct_text(text):
    if not isinstance(text, str):
        return ""

    '''
    if len(text.strip()) == 0 or not detect_language(text):
        return ""
    '''
    text = text.lower()  # Μετατροπή σε πεζά
    text = re.sub(r'\b(lol)+\b', 'lol', text)  
    text = re.sub(r'\b(ha)+\b', 'ha', text)  
    for word, replacement in corrections.items():
        text = re.sub(rf"\b{re.escape(word)}\b", replacement, text) 

    return text

custom_stopwords = {
    "to","I","the","a","my","i","and","is","in","for","of","it","on","have","you","so",
    "me","but","that","not","you","with","be","im","now","IM","amp","up","go","get","this",
    "with","just","I'm","was","at","be","out","all","are","work","now","got","do","day","back",
    "your","from"
}


def preprocess_text(text):
    if not isinstance(text, str):  
        return ""  
    
    
    #text = text.lower()  
    text = correct_text(text)
    text = re.sub(r"http\S+", "", text) #urls
    text = re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', text)
    text = re.sub(r"\d+", "", text)  #numbers
    text = re.sub(r"[^\w\s]", "", text)  #simeia stiksis
    text = re.sub(r"\s+", " ", text).strip()  #kena
    

    '''
    for punct in sentiment_punctuation:
        text = text.replace(punct, f" {punct} ")  
    text = re.sub(r"[^\w\s!?…:-]", "", text)
    '''

    #blob = TextBlob(text)
    
    #tokens = word_tokenize(text)  
    #tokens = [word for word in tokens if word not in stop_words]  
    #tokens = [word for word in tokens if len(word) > 2]  
    #tokens = [lemmatizer.lemmatize(word) for word in tokens] 
    tokens = text.split()  
    tokens = [word for word in tokens if word not in custom_stopwords] 
    tokens = [stemmer.stem(word) for word in tokens]

    
    
    return " ".join(tokens)


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



df["text"] = df["text"].apply(preprocess_text)
df_test["text"] = df_test["text"].apply(preprocess_text)
df_val["text"] = df_val["text"].apply(preprocess_text)



tokenized_texts = [text.split() for text in df["text"]]
model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4, sg=1)


#splitting the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
#splitting the training data set to validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

def get_average_embedding(text, model, vector_size):
    words = text.split()
    valid_words = [word for word in words if word in model.wv]
    
    if not valid_words:
        return np.zeros(vector_size)
    
    embeddings = [model.wv[word] for word in valid_words]
    return np.mean(embeddings, axis=0)

vector_size = 100  

X_train_embed = np.array([get_average_embedding(text, model, vector_size) for text in X_train])

X_val_embed = np.array([get_average_embedding(text, model, vector_size) for text in X_val])

X_test_embed = np.array([get_average_embedding(text, model, vector_size) for text in X_test])


X_train_tensor = torch.tensor(X_train_embed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

X_val_tensor = torch.tensor(X_val_embed, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_embed, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

input_dim = vector_size

hidden_dim = 150

output_dim = len(set(df["label"]))

model_nn = SentimentClassifier(input_dim, hidden_dim, output_dim)

#Optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.01)


#tried 20,30,40,45,50,55 best one 50
num_epochs=60
train_losses = []
val_accuracies = []
#best_val_acc = 0
#patience_counter = 0
#patience = 5


for epoch in range(num_epochs):
    model_nn.train()
    optimizer.zero_grad()
    
    outputs = model_nn(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())

    # Validation accuracy
    model_nn.eval()
    with torch.no_grad():
        val_outputs = model_nn(X_val_tensor)
        val_preds = torch.argmax(val_outputs, dim=1)
        val_acc = (val_preds == y_val_tensor).float().mean().item()
        val_accuracies.append(val_acc)
        #if val_acc > best_val_acc:
        #    best_val_acc = val_acc
        #    patience_counter = 0
        #    best_model_wts = model_nn.state_dict()  
        #else:
        #    patience_counter += 1
    
        #if patience_counter >= patience:
        #    print(f"Early stopping at epoch {epoch+1}")
        #break
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | Val Accuracy: {val_acc:.4f}")

#model_nn.load_state_dict(best_model_wts)

model_nn.eval()
with torch.no_grad():
    test_outputs = model_nn(X_test_tensor)
    test_preds = torch.argmax(test_outputs, dim=1)
    test_acc = (test_preds == y_test_tensor).sum().item() / len(y_test_tensor)

print("Learning Rate:",0.01);
print("Hidden Dim:",hidden_dim);
print("Epochs:", num_epochs);
print(f"Test Accuracy: {test_acc:.4f}")

#accuracy,precision,recall
print("Classification Report:")
#print(classification_report(y_test, test_preds.numpy(), digits=4))
print(classification_report(y_test_tensor, test_preds.numpy(), digits=4))

#y_test["predicted_label"] = model_nn.predict(X_test)
#y_test_tensor["predicted_label"] = model_nn.predict(X_test_tensor)
#y_test_output = y_test[["ID", "predicted_label"]]
#y_test_output.to_csv("/home/erginadimitraina/AI2/submission2.csv", index=False)

# Δημιουργία DataFrame για αποθήκευση προβλέψεων
#df_test_output = pd.DataFrame({
#    "ID": df_test.index, 
#    "predicted_label": test_preds.numpy()
#})
#df_test_output.to_csv("/home/erginadimitraina/AI2/submission2.csv", index=False)

'''
df_test_output = pd.DataFrame({
    "ID": X_test_tensor,
    "predicted_label": test_preds.numpy()
})
'''
df_test_embed = np.array([get_average_embedding(text, model, vector_size) for text in df_test["text"]])
df_test_tensor = torch.tensor(df_test_embed, dtype=torch.float32)

# Predict
model_nn.eval()
with torch.no_grad():
    outputs = model_nn(df_test_tensor)
    predictions = torch.argmax(outputs, dim=1)

# Δημιουργία DataFrame για το submission
df_submission = pd.DataFrame({
    "ID": df_test["ID"],  # προϋπόθεση: df_test έχει στήλη ID
    "predicted_label": predictions.numpy()
})

df_submission.to_csv("/home/erginadimitraina/AI2/submission2.csv", index=False)


#PLOTS

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
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


conf_mat = confusion_matrix(y_test, test_preds.numpy())


plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#ROC Curve
model_nn.eval()
with torch.no_grad():
    y_outputs = model_nn(X_test_tensor)  # Get probabilities
    y_scores = y_outputs[:, 1].numpy()
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

