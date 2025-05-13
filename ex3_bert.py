import nltk
from random import seed
import random
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from nltk.corpus import opinion_lexicon,words
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer,BertForSequenceClassification,get_linear_schedule_with_warmup
import torch 
from torch.utils.data import DataLoader, TensorDataset,RandomSampler,SequentialSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

nltk.download('opinion_lexicon')

df = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/train_dataset.csv")
df.rename(columns={"Text": "text", "Label": "label"}, inplace=True)

df_test = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/test_dataset.csv")
df_test.rename(columns={"Text": "text"}, inplace=True)

df_val = pd.read_csv("/home/erginadimitraina/AI2/ai-2-deep-learning-for-nlp-homework-1/val_dataset.csv")
df_val.rename(columns={"Text": "text", "Label": "label"}, inplace=True)

#για δοκιμη επειδη αν παρω ολοκληρο το σετ κανει πανω απο 20 ωρες run time
df = df.sample(frac=0.05, random_state=42)

df.rename(columns={"Text": "text", "Label": "label"}, inplace=True)
df_test.rename(columns={"Text": "text"}, inplace=True)
df_val.rename(columns={"Text": "text", "Label": "label"}, inplace=True)

def preprocess_text(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text



df["text"] = df["text"].apply(preprocess_text)
df_test["text"] = df_test["text"].apply(preprocess_text)
df_val["text"] = df_val["text"].apply(preprocess_text)

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)


#WORD CLOUD PLOT
'''
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

# Training data
#Reviews = "[CLS] " +train_df['Cleaned_sentence'] + "[SEP]"
#Reviews = df['text']
#Target = df['label']


# Test data
#test_reviews =  "[CLS] " +test_df['Cleaned_sentence'] + "[SEP]"
#test_reviews = df_test['text']
#test_targets = df_test['label']

#df = df.dropna(subset=["label"])
#df["label"] = df["label"].astype(int)


#splitting the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
#splitting the training data set to validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)



#Tokenize and encode the data using the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)

def encode_data(tokenizer, texts, labels=None, max_len=128):
    encoding = tokenizer.batch_encode_plus(
        texts.tolist(),
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    if labels is not None:
        return TensorDataset(encoding['input_ids'], encoding['attention_mask'], torch.tensor(labels.tolist()))
    else:
        return TensorDataset(encoding['input_ids'], encoding['attention_mask'])

train_dataset = encode_data(tokenizer, X_train, y_train)
val_dataset = encode_data(tokenizer, X_val, y_val)

train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=4)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs=2
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs)

# Evaluation metrics
def compute_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return {
        'accuracy': accuracy_score(labels_flat, preds_flat),
        'precision': precision_score(labels_flat, preds_flat, average='weighted'),
        'recall': recall_score(labels_flat, preds_flat, average='weighted'),
        'f1': f1_score(labels_flat, preds_flat, average='weighted')
    }


# Evaluation loop
def evaluate(model, dataloader):
    model.eval()
    loss_total = 0
    predictions, true_vals = [], []

    for batch in dataloader:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        loss_total += loss.item()

        predictions.append(logits.detach().cpu().numpy())
        true_vals.append(inputs['labels'].cpu().numpy())

    preds = np.concatenate(predictions, axis=0)
    labels = np.concatenate(true_vals, axis=0)
    return loss_total / len(dataloader), preds, labels


# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    print(f"\nEpoch {epoch + 1}/{epochs}")
    for batch in tqdm(train_loader):
        batch = tuple(b.to(device) for b in batch)
        model.zero_grad()

        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Training loss: {avg_train_loss:.4f}")

    val_loss, preds, labels = evaluate(model, val_loader)
    metrics = compute_metrics(preds, labels)
    print(f"Validation loss: {val_loss:.4f}")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

# Test predictions
def predict_on_test(model, tokenizer, texts):
    model.eval()
    test_dataset = encode_data(tokenizer, texts)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)

    predictions = []
    for batch in test_loader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(input_ids=batch[0], attention_mask=batch[1])
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
    return predictions

# Predict on test and export
test_preds = predict_on_test(model, tokenizer, df_test['text'])
submission = pd.DataFrame({'Id': df_test.index, 'Predicted': test_preds})
#submission.to_csv('submission3.csv', index=False)
#print("Submission saved to submission3.csv")



'''
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2,output_attentions = False,output_hidden_states = False)

batch_size = 4

dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size
)

dataloader_val = DataLoader(
    dataset_val,
    sampler=RandomSampler(dataset_val),
    batch_size=32
)




optimizer = AdamW(model.parameters(),lr = 1e-5,eps = 1e-8)

#Devlin et al. recommend only 2-4 epochs of training for fine-tuning BERT on a specific NLP task
epochs = 10

scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps = len(dataloader_train)*epochs)

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')

def accuracy_per_class(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        correct = np.sum(y_preds == label)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0
        print(f'Class: {label}')
        print(f'Accuracy: {correct}/{total} ({accuracy:.2f})\n')

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(device)


def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

for epoch in tqdm(range(1, epochs+1)):
    model.train()
    loss_train_total = 0
    
    progress_bar = tqdm(dataloader_train, 
                        desc='Epoch {:1d}'.format(epoch), 
                        leave=False, 
                        disable=False)
    
    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }
        
        outputs = model(**inputs)
        loss = outputs[0]
        loss_train_total +=loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})     
    
    #torch.save(model.state_dict(), f'Models/BERT_ft_Epoch{epoch}.model')
    
    tqdm.write('\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (weighted): {val_f1}')


accuracy_per_class(predictions, true_vals)


max_len= 128
#Tokenize and encode the sentences
X_train_encoded = tokenizer.batch_encode_plus(Reviews.tolist(),padding=True, truncation=True,max_length = max_len,return_tensors='pt')

X_val_encoded = tokenizer.batch_encode_plus(X_val.tolist(), padding=True, truncation=True,max_length = max_len,return_tensors='pt')

X_test_encoded = tokenizer.batch_encode_plus(X_test.tolist(), padding=True, truncation=True,max_length = max_len,return_tensors='pt')

k = 0
print('Training Comments -->>',Reviews[k])
print('\nInput Ids -->>\n',X_train_encoded['input_ids'][k])
print('\nDecoded Ids -->>\n',tokenizer.decode(X_train_encoded['input_ids'][k]))
print('\nAttention Mask -->>\n',X_train_encoded['attention_mask'][k])
print('\nLabels -->>',Target[k])


y_val = y_val.to_numpy() if hasattr(y_val, "to_numpy") else y_val
Target = Target.to_numpy() if hasattr(Target, "to_numpy") else Target
y_test = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test

# Intialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Compile the model with an appropriate optimizer, loss function, and metrics
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
def accuracy(preds, labels):
    pred_labels = torch.argmax(preds, dim=1)  # Πάρε την κατηγορία με τη μεγαλύτερη πιθανότητα
    return torch.sum(pred_labels == labels).item() / labels.size(0)  # Υπολογισμός ακρίβειας
train_dataset = TensorDataset(
    X_train_encoded['input_ids'],
    X_train_encoded['attention_mask'],
    torch.tensor(Target, dtype=torch.long)
)
val_dataset = TensorDataset(
    X_val_encoded['input_ids'],
    X_val_encoded['attention_mask'],
    torch.tensor(y_val, dtype=torch.long)
)
test_dataset = TensorDataset(
    X_test_encoded['input_ids'],
    X_test_encoded['attention_mask'],
    torch.tensor(y_test, dtype=torch.long)
)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Απλή accuracy function
def accuracy(preds, labels):
    pred_labels = torch.argmax(preds, dim=1)
    return (pred_labels == labels).sum().item() / labels.size(0)

# Loss και optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0

        for batch in train_dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            acc = accuracy(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += acc

        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_accuracy / len(train_dataloader)
        print(f"\nEpoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f}")

        evaluate_model(model, val_dataloader)

# Evaluation loop (για validation)
def evaluate_model(model, val_dataloader):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            acc = accuracy(logits, labels)

            total_loss += loss.item()
            total_accuracy += acc

    avg_loss = total_loss / len(val_dataloader)
    avg_accuracy = total_accuracy / len(val_dataloader)
    print(f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {avg_accuracy:.4f}")

# Final test
def test_model(model, test_dataloader):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            acc = accuracy(logits, labels)

            total_loss += loss.item()
            total_accuracy += acc

    avg_loss = total_loss / len(test_dataloader)
    avg_accuracy = total_accuracy / len(test_dataloader)
    print(f"\nTest Loss: {avg_loss:.4f} | Test Accuracy: {avg_accuracy:.4f}")

# Τρέξιμο
train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, num_epochs=3)
test_model(model, test_dataloader)
'''