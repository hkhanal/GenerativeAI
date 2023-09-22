import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification

import torch.optim as optim

class TextClassificationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Example data
data = [
    {
        "inputs": "Messi scored 5 goals in United States soccer league",
        "label": "Soccer Player"
    },
    # Add more data entries here
]

dataset = TextClassificationDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))

# Define the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        inputs = tokenizer(batch['inputs'], padding=True, truncation=True, return_tensors='pt', max_length=128)
        labels = torch.tensor([labels.index(entry['label']) for entry in batch])

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

# Save the trained model
model.save_pretrained('text_classification_model')

