import tensorflow as tf

def build_bi_lstm_model(vocab_size, embedding_dim=300, lstm_units=128, max_length=50):
    """
    Builds a Bidirectional LSTM model for text classification.
    
    Parameters:
    - vocab_size: Size of the vocabulary.
    - embedding_dim: Dimension of the embedding layer.
    - lstm_units: Number of units in the LSTM layer.
    - max_length: Maximum length of input sequences.
    
    Returns:
    - model: A compiled Keras model.
    """
    
    input = tf.keras.Input(shape=(max_length,), name="input_ids")

    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding_layer"
    )(input)

    context = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True),
        name="bidirectional_lstm"
    )(embedding)

    output = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(vocab_size, activation='softmax'),
        name="output_layer"
    )(context)

    model = tf.keras.Model(inputs=input, outputs=output, name="BiLSTM_LanguageModel")
    return model