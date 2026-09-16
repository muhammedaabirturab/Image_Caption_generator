"""
Encoder-decoder captioning model.

    Image feature (2048)                Caption tokens (max_len)
          |                                      |
     Dense + Dropout                    Embedding + Dropout
     (image embedding)                        |
          |                                  LSTM
          |                                    |
          +-------------- add -----------------+
                          |
                    Dense (relu)
                          |
                Dense (softmax over vocab)
                          |
                      next word

The CNN image feature and the LSTM's summary of the caption-so-far are
projected into the same-sized space and merged by addition ("merge
architecture"), then passed through a small classifier head that
predicts the next word. This keeps the model simple and explainable:
there is exactly one recurrent layer and one place where the two
modalities meet.
"""

from tensorflow.keras.layers import (
    Add, Dense, Dropout, Embedding, Input, LSTM
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src import config


def build_captioning_model(vocab_size, max_length,
                            embedding_dim=config.EMBEDDING_DIM,
                            lstm_units=config.LSTM_UNITS,
                            dropout_rate=config.DROPOUT_RATE,
                            learning_rate=config.LEARNING_RATE):
    # Image feature branch
    image_input = Input(shape=(config.CNN_FEATURE_DIM,), name="image_features")
    image_dropout = Dropout(dropout_rate)(image_input)
    image_dense = Dense(embedding_dim, activation="relu", name="image_embedding")(image_dropout)

    # Caption (text) branch
    caption_input = Input(shape=(max_length,), name="caption_input")
    caption_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True,
                                   name="word_embedding")(caption_input)
    caption_dropout = Dropout(dropout_rate)(caption_embedding)
    caption_lstm = LSTM(lstm_units, name="decoder_lstm")(caption_dropout)

    # Merge image and text representations
    merged = Add(name="merge_image_text")([image_dense, caption_lstm])
    decoder_dense = Dense(lstm_units, activation="relu", name="decoder_dense")(merged)
    output = Dense(vocab_size, activation="softmax", name="next_word")(decoder_dense)

    model = Model(inputs=[image_input, caption_input], outputs=output,
                  name="image_caption_generator")
    # Targets are plain integer word indices (not one-hot) so that batches
    # of a several-thousand-word vocabulary stay memory-efficient.
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model
