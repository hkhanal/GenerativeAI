#!/usr/bin/env python

import torch
from torch import nn
import os
import glob
import torch
from torch import nn
import zipfile
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas  as pd
from sklearn.model_selection import train_test_split

#Load pre-trained BERT model and tokenizer
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_path = "./data"
# Custom Dataset Class for Pre-Encoded Data
class EncodedTextDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }


# Function to Encode the Texts
def encode_texts(texts, labels, tokenizer, max_length=512):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_text['input_ids'].flatten())
        attention_masks.append(encoded_text['attention_mask'].flatten())

    return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(labels)


# Build the classification model
class TextClassifier(nn.Module):
    def __init__(self, bert_model):
        super(TextClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

model = TextClassifier(bert_model).to(device)

# load training data
df = pd.read_csv(f'{data_path}/train_essays.csv')  # Read CSV file into a DataFrame
df['label'] = df.generated.copy()

ex_files = ["train_drcat_01.csv", "train_drcat_02.csv", "train_drcat_03.csv", "train_drcat_04.csv"]
df_extra = pd.concat(pd.read_csv(os.path.join(data_path, f)) for f in ex_files)

df_data = pd.concat([df_extra[["text", "label"]], df[["text", "label"]]])
# Encode texts and labels

"""
texts = df_data["text"].values.tolist()
labels = df_data["label"].values.tolist()
input_ids, attention_masks, labels = encode_texts(texts, labels, tokenizer)
torch.save({
    'input_ids': input_ids,
    'attention_masks': attention_masks,
    'labels': labels
}, os.path.join(data_path, 'tensor_data_llm_fake.pth'))
"""

# Load the tensors from the file
saved_tensors = torch.load(os.path.join(data_path, 'tensor_data_llm_fake.pth'))

input_ids = saved_tensors['input_ids']
attention_masks = saved_tensors['attention_masks']
labels = saved_tensors['labels']

# Split the data into training and testing sets
train_ids, test_ids, train_masks, test_masks, train_labels, test_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.2
)

# Create Dataset and DataLoader for training and testing
train_dataset = EncodedTextDataset(train_ids, train_masks, train_labels)
test_dataset = EncodedTextDataset(test_ids, test_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Training and Evaluation
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = nn.BCEWithLogitsLoss()


def evaluate(model, data_loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            predictions = torch.round(torch.sigmoid(outputs.squeeze()))
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Prediction Function
def predict(texts, model, tokenizer, max_length=512):
    model.eval()
    predictions = []

    with torch.no_grad():
        for text in texts:
            encoded_text = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            ).to(device)

            input_ids = encoded_text['input_ids']
            attention_mask = encoded_text['attention_mask']
            output = model(input_ids, attention_mask)
            prediction = torch.sigmoid(output).item()
            predictions.append(prediction)

    return predictions

print ("Start training model")
epoches =6
for epoch in range(epoches):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

    # Evaluate on the testing set
    test_accuracy = evaluate(model, test_loader)
    print(f'Epoch {epoch + 1}, Test Accuracy: {test_accuracy:.2f}')

# Save the Model
model_dir = "./model_dir"
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load the Model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
model_path = os.path.join(model_dir, "bert_text_classifier.pth")
save_model(model, model_path)

#submission 
# Read the test dataset from a CSV file
test = pd.read_csv(os.path.join(data_dir, "test_essays.csv"))
texts_sub = test["text"].values.tolist()

# Generate predictions for the test dataset using the trained model

print ("Making prediction")
                   
test["generated"] = predict(texts_sub, model, tokenizer, max_length=512)
# Create a submission dataframe with the required columns
submission = test[["id", "generated"]]
# Save the submission dataframe to a CSV file
submission.to_csv("submission.csv", index=False)
# Display the first few rows of the submission dataframe
submission.head()

