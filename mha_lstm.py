import tensorflow as tf

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
        
        return context_vector, attention_weights

# Example usage:
#input_sequence = tf.keras.layers.Input(shape=(sequence_length, input_dim))
#lstm_units = 64  # Number of units in the LSTM layer
#attention_layer = AttentionWithLSTM(units=lstm_units)
#context_vector, attention_weights = attention_layer(input_sequence)