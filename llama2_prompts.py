import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

class TextClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input_text"]
        label = item["label"]

        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        label_id = self.tokenizer.encode(label, add_special_tokens=False)[0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_id": label_id
        }
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/llama2-base-uncased")
max_length = 128

train_dataset = TextClassificationDataset(dataset, tokenizer, max_length)
model = AutoModelForSequenceClassification.from_pretrained("Helsinki-NLP/llama2-base-uncased", num_labels=num_labels)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=None,
    train_dataset=train_dataset,
)

trainer.train()