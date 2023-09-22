import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

class TextClassificationDataset(tf.data.Dataset):
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

# Create a TensorFlow dataset
dataset = TextClassificationDataset(data)


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))

# Create a custom Keras model
class TextClassificationModel(tf.keras.Model):
    def __init__(self, bert_model):
        super(TextClassificationModel, self).__init__()
        self.bert = bert_model

    def call(self, inputs, training=False):
        outputs = self.bert(inputs, training=training)
        return outputs.logits

# Instantiate the custom model
text_classifier = TextClassificationModel(model)

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in dataset:
        inputs = tokenizer(batch['inputs'], padding=True, truncation=True, return_tensors='tf', max_length=128)
        labels = tf.convert_to_tensor([labels.index(entry['label'])])

        with tf.GradientTape() as tape:
            logits = text_classifier(inputs, training=True)
            loss = loss_fn(labels, logits)

        grads = tape.gradient(loss, text_classifier.trainable_variables)
        optimizer.apply_gradients(zip(grads, text_classifier.trainable_variables))

        total_loss += loss.numpy()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataset)}")

# Save the trained model
text_classifier.save_weights('text_classification_model')

