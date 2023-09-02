import tensorflow as tf

class MultiHeadAttentionWithLSTM(tf.keras.layers.Layer):
    def __init__(self, num_heads, units):
        super(MultiHeadAttentionWithLSTM, self).__init__()
        self.num_heads = num_heads
        self.units = units
        assert units % num_heads == 0, "units must be divisible by num_heads"
        self.depth = units // num_heads
        
        self.attention_heads = [AttentionWithLSTM(self.depth) for _ in range(num_heads)]
        
    def call(self, inputs):
        attention_outputs = [head(inputs) for head in self.attention_heads]
        attention_outputs = tf.stack(attention_outputs, axis=-1)  # Stack the outputs from different heads
        attention_outputs = tf.keras.layers.Flatten()(attention_outputs)  # Flatten the stacked outputs
        
        return attention_outputs

class AttentionWithLSTM(tf.keras.layers.Layer):
    def __init__(self, units):
        super(AttentionWithLSTM, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        # Define the LSTM layer
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True)
        self.w1 = tf.keras.layers.Dense(self.units)
        self.w2 = tf.keras.layers.Dense(self.units)
        self.v = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        # Split the input into keys and values
        keys = self.lstm(inputs)
        values = self.lstm(inputs)
        
        # Calculate attention scores
        query = self.w1(inputs)
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.v(tf.nn.tanh(query_with_time_axis + self.w2(keys)))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Multiply attention weights with values to get context vector
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector

# Create the custom model
def create_custom_model(input_dim, sequence_length, embedding_dim, num_heads, lstm_units):
    input_sequence = tf.keras.layers.Input(shape=(sequence_length,))
    
    # Embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim, embedding_dim)(input_sequence)
    
    # Multi-head attention layer with LSTM
    multi_head_attention_layer = MultiHeadAttentionWithLSTM(num_heads=num_heads, units=lstm_units)(embedding_layer)
    
    # LSTM layers on top
    lstm_layer = tf.keras.layers.LSTM(lstm_units)(multi_head_attention_layer)
    
    # Output layer (customize for your specific task)
    output = tf.keras.layers.Dense(output_dim, activation='softmax')(lstm_layer)
    
    model = tf.keras.models.Model(inputs=input_sequence, outputs=output)
    return model

# Example usage:
input_dim = 10000  # Vocabulary size
sequence_length = 50
embedding_dim = 128
num_heads = 4
lstm_units = 64
output_dim = 10  # Adjust for your specific task

custom_model = create_custom_model(input_dim, sequence_length, embedding_dim, num_heads, lstm_units)
custom_model.summary()