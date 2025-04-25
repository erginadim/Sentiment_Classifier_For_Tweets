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

#reproducibility
random.seed(42)
np.random.seed(42)

stemmer = PorterStemmer()
#load datasets
df = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/train_dataset.csv")
df.rename(columns={"Text": "text", "Label": "label"}, inplace=True)

df_test = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/test_dataset.csv")
df_test.rename(columns={"Text": "text"}, inplace=True)

df_val = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/val_dataset.csv")
df_val.rename(columns={"Text": "text", "Label": "label"}, inplace=True)


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
    "ha": "laugh"  
    }



def correct_text(text):
    if not isinstance(text, str):
        return ""

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
    text = re.sub(r"\d+", "", text)  #numbers
    text = re.sub(r"[^\w\s]", "", text)  #simeia stiksis
    text = re.sub(r"\s+", " ", text).strip()  #kena
    '''
    for punct in sentiment_punctuation:
        text = text.replace(punct, f" {punct} ")  
    text = re.sub(r"[^\w\s!?…:-]", "", text)
    '''
    #tokens = word_tokenize(text)  
    #tokens = [word for word in tokens if word not in stop_words]  
    #tokens = [word for word in tokens if len(word) > 2]  
    #tokens = [lemmatizer.lemmatize(word) for word in tokens] 
    tokens = text.split()  
    tokens = [word for word in tokens if word not in custom_stopwords] 
    tokens = [stemmer.stem(word) for word in tokens]

    
    
    return " ".join(tokens)



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
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return self.softmax(out)

input_dim = vector_size
hidden_dim = 128
output_dim = len(set(df["label"]))

model_nn = SentimentClassifier(input_dim, hidden_dim, output_dim)

#predictions
#y_pred = model_nn.predict(X_test_tensor)

#evaluation 
#accuracy = accuracy_score(y_test,y_pred)
#print(f"Accuracy: {accuracy: .2f}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)

#tried 20,30,40,45,50,55 best one 50
num_epochs=50

for epoch in range(num_epochs):
    model_nn.train()
    
    optimizer.zero_grad()
    outputs = model_nn(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation
    model_nn.eval()
    with torch.no_grad():
        val_outputs = model_nn(X_val_tensor)
        val_preds = torch.argmax(val_outputs, dim=1)
        val_acc = (val_preds == y_val_tensor).sum().item() / len(y_val_tensor)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_acc:.4f}")

model_nn.eval()
with torch.no_grad():
    test_outputs = model_nn(X_test_tensor)
    test_preds = torch.argmax(test_outputs, dim=1)
    test_acc = (test_preds == y_test_tensor).sum().item() / len(y_test_tensor)

print(f"Test Accuracy: {test_acc:.4f}")

#class SentimentClassifier(object):
    
  # constructor
 # def __init__(self, x, y, z):
  #  self.x = x
   # self.y = y
   # self.z = z

  # calculate the forward path
  #def forward(self):
   # self.a = self.relu(self.x)
   # self.b = self.y()
   # self.f = self.a * self.b
   # return f
