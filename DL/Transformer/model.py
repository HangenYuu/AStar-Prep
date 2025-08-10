# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: model.py
# @Software: PyCharm
import numpy as np
from keras.layers import (
    Dense,
    Input,
    Dropout,
    Embedding,
    Flatten,
    MaxPooling1D,
    Conv1D,
    SimpleRNN,
    LSTM,
    GRU,
    Multiply,
    GlobalMaxPooling1D,
)
from keras.layers import (
    Bidirectional,
    Activation,
    BatchNormalization,
    GlobalAveragePooling1D,
    MultiHeadAttention,
)
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import concatenate
import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import (
    Input,
    Embedding,
    Convolution1D,
    MaxPooling1D,
    Concatenate,
    Dropout,
    GlobalMaxPooling1D,
    MaxPooling2D,
    GlobalMaxPooling2D,
    SimpleRNN,
)
from keras.layers import (
    Flatten,
    Dense,
    Activation,
    BatchNormalization,
    CuDNNGRU,
    CuDNNLSTM,
    LSTM,
    GRU,
)
from keras.models import Model, Sequential
from keras.regularizers import l2
from torch.nn import TransformerEncoder
from keras.optimizers import Adam, SGD, Adagrad
from keras.layers import (
    Bidirectional,
    Activation,
    BatchNormalization,
    GlobalAveragePooling1D,
    MultiHeadAttention,
)
from keras_nlp.layers import PositionEmbedding, SinePositionEncoding

# from transformer import TransformerEncoder,PositionalEmbedding,SinCosPositionEmbedding,Attention


def qq():

    inx3 = tf.keras.layers.Input(shape=train_embeding_data.shape[1:])
    x3 = inx3
    x3 = tf.keras.layers.Embedding(input_dim=21, output_dim=512, mask_zero=0)
    x3_pos = SinePositionEncoding(max_wavelength=50)(x3)
    x3 = x3 + x3_pos
    x3 = keras_nlp.layers.TransformerDecoder(intermediate_dim=128, num_heads=10)(x3)
    x3 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=512, return_sequences=True)
    )(x3)
    model = tf.keras.layers.GlobalAveragePooling1D()(x3)
    return model


class TransformerEncoder1(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dense_dim": self.dense_dim,
            }
        )
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "sequence_length": self.sequence_length,
                "input_dim": self.input_dim,
            }
        )
        return config


def transformer(
    length,
    out_length,
):
    inputs = Input(
        name="inputs",
        shape=[
            1,
            length,
        ],
        dtype="float64",
    )
    # x = Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True)(inputs)
    x = TransformerEncoder1(20, 32, 4)(inputs)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(out_length, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


# from tensorflow.keras.layers import MultiHeadAttention, Dropout, LayerNormalization
def PositionalEmbedding_Transformer(length, embed_dim, top_words, out_length):
    inputs = Input(
        name="inputs",
        shape=[
            length,
        ],
        dtype="float64",
    )
    x = PositionalEmbedding(
        sequence_length=length, input_dim=top_words, output_dim=embed_dim
    )(inputs)
    x = TransformerEncoder1(embed_dim, 32, 4)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(out_length, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


def base(
    length,
    out_length,
):

    ed = ["embedding_dimension"]
    ps = ["pool_size"]
    fd = ["fully_dimension"]
    dp = ["drop_out"]
    lr = ["learning_rate"]
    l2value = 0.001

    main_input = Input(shape=(length,), dtype="int64", name="main_input")

    x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)

    a = Convolution1D(
        64, 2, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(x)
    apool = MaxPooling1D(pool_size=ps, strides=1, padding="same")(a)

    b = Convolution1D(
        64, 3, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(x)
    bpool = MaxPooling1D(pool_size=ps, strides=1, padding="same")(b)

    c = Convolution1D(
        64, 8, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(x)
    cpool = MaxPooling1D(pool_size=ps, strides=1, padding="same")(c)

    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Flatten()(merge)

    x = Dense(fd, activation="relu", name="FC1", kernel_regularizer=l2(l2value))(x)

    outputs = Dense(
        out_length, activation="sigmoid", name="outputs", kernel_regularizer=l2(l2value)
    )(x)

    model = Model(inputs=main_input, outputs=outputs)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

    # model.summary()

    return model


# def base_sunhao(length, out_length, ):
#     ed = ['embedding_dimension']    #100
#     ps = ['pool_size']              #5
#     fd = ['fully_dimension']        #128
#     dp = ['drop_out']               #0.5
#     lr = ['learning_rate']          #0.001
#     l2value = 0.001
#
#     inputs = Input(name='inputs', shape=[length, ], dtype='int64')
#
#     x = PositionalEmbedding(sequence_length=length, input_dim=21, output_dim=ed)(inputs)
#     a = TransformerEncoder(ed, 32, 4)(x)
#
#     #a = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a)
#
#     #******************************************
#
#     a_1 = Convolution1D(64, 2, activation='relu', padding='same', kernel_regularizer=l2(l2value))(a)
#     apool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(a_1)
#
#     b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(a)
#     bpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)
#
#     c = Convolution1D(64, 8, activation='relu', padding='same', kernel_regularizer=l2(l2value))(a)
#     cpool = MaxPooling1D(pool_size=ps, strides=1, padding='same')(c)
#
#
#     merge = Concatenate(axis=-1)([apool, bpool, cpool, a])
#     merge = Dropout(dp)(merge)
#
#     #x = CuDNNGRU(100, return_sequences=True)(merge)
#     x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)
#
#     x = MultiHeadAttention(num_heads=2,key_dim=ed)(x,x,x)
#
#     x = Flatten()(x)
#
#     x = Dense(fd, activation='relu', name='FC1', kernel_regularizer=l2(l2value))(x)
#
#     #******************************************
#
#     outputs = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)
#     model = Model(inputs, outputs)
#     adam = Adam(learning_rate=lr)
#     model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#
#     model.summary()
#
#     return model
# {'embedding_dimension': 100, 'pool_size': 5, 'fully_dimension': 128, 'drop_out': 0.5, 'learning_rate': 0.001}
def BiGRU_base(length, length3, out_length, pool_size=5):
    print()

    ps = pool_size
    fd = 128  # 128

    l2value = 0.001

    main_input = Input(shape=(length, length3), name="input")

    # x = Embedding(output_dim=ed, input_dim=21, input_length=length)(main_input)

    a = Convolution1D(
        64, 2, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(main_input)
    apool = MaxPooling1D(
        pool_size=ps, strides=1, padding="same", data_format="channels_first"
    )(a)

    b = Convolution1D(
        64, 3, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(main_input)
    bpool = MaxPooling1D(
        pool_size=ps, strides=1, padding="same", data_format="channels_first"
    )(b)

    c = Convolution1D(
        64, 8, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(main_input)
    cpool = MaxPooling1D(
        pool_size=ps, strides=1, padding="same", data_format="channels_first"
    )(c)
    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    # merge = Dropout(dp)(merge)

    x = Bidirectional(CuDNNGRU(50, return_sequences=True))(merge)
    x = Flatten()(x)
    x = Dense(fd, activation="relu", name="FC1", kernel_regularizer=l2(l2value))(x)
    # outputs = Dense(out_length, activation='sigmoid', name='outputs')(x)
    outputs = Dense(
        out_length, activation="sigmoid", name="outputs", kernel_regularizer=l2(l2value)
    )(x)

    model = Model(inputs=main_input, outputs=outputs)

    # model.summary()

    return model


def MPMABP(
    length,
    out_length,
):
    ed = ["embedding_dimension"]  # 100
    ps = ["pool_size"]  # 5
    fd = ["fully_dimension"]  # 128
    dp = ["drop_out"]  # 0.5
    lr = ["learning_rate"]  # 0.001
    l2value = 0.001

    main_input = Input(
        shape=(
            1,
            length,
        ),
        dtype="int64",
        name="main_input",
    )

    # x = Embedding(output_dim=ed, input_dim=21, input_length=length, embeddings_initializer='uniform')(main_input)
    x = keras.layers.BatchNormalization()(main_input)

    a = Convolution1D(
        64, 3, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(x)
    a = Bidirectional(CuDNNLSTM(32, return_sequences=True))(a)
    a = keras.layers.LeakyReLU(alpha=0.3)(a)
    apool = MaxPooling1D(pool_size=3, strides=1, padding="same")(a)

    b = Convolution1D(
        64, 5, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(x)
    b = Bidirectional(CuDNNLSTM(32, return_sequences=True))(b)
    b = keras.layers.LeakyReLU(alpha=0.3)(b)
    bpool = MaxPooling1D(pool_size=3, strides=1, padding="same")(b)

    c = Convolution1D(
        64, 8, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(x)
    c = Bidirectional(CuDNNLSTM(32, return_sequences=True))(c)
    c = keras.layers.LeakyReLU(alpha=0.3)(c)
    cpool = MaxPooling1D(pool_size=3, strides=1, padding="same")(c)

    d = Convolution1D(
        64, 10, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(x)
    d = keras.layers.LeakyReLU(alpha=0.3)(d)
    d = Bidirectional(CuDNNLSTM(32, return_sequences=True))(d)
    dpool = MaxPooling1D(pool_size=3, strides=1, padding="same")(d)

    e = Convolution1D(
        64, 12, activation="relu", padding="same", kernel_regularizer=l2(l2value)
    )(x)
    e = keras.layers.LeakyReLU(alpha=0.3)(e)
    e = Bidirectional(CuDNNLSTM(32, return_sequences=True))(e)
    epool = MaxPooling1D(pool_size=3, strides=1, padding="same")(e)

    # f = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)

    # x_batchnorm = keras.layers.BatchNormalization()(x)
    # cnnrnn = Concatenate(axis=-1)([apool, bpool, cpool, dpool, epool, x, f])
    cnnrnn = Concatenate(axis=-1)([apool, bpool, cpool, dpool, epool, x])

    CNNRNN = Flatten()(cnnrnn)
    CNNRNN = Dense(
        64, activation="relu", name="dense1", kernel_regularizer=l2(l2value)
    )(CNNRNN)
    CNNRNN = Dropout(dp)(CNNRNN)
    CNNRNN = Dense(
        128, activation="relu", name="dense2", kernel_regularizer=l2(l2value)
    )(CNNRNN)

    output = Dense(
        out_length, activation="sigmoid", name="output", kernel_regularizer=l2(l2value)
    )(CNNRNN)

    model = Model(inputs=main_input, outputs=output)
    adam = Adagrad(learning_rate=lr)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

    # model.summary()

    return model


def transformer1(
    length,
    length3,
    out_length,
):
    ed = ["embedding_dimension"]  # 100
    ps = ["pool_size"]  # 5
    fd = ["fully_dimension"]  # 128
    dp = ["drop_out"]  # 0.5
    lr = ["learning_rate"]  # 0.001
    l2value = 0.001

    # inputs = Input(name='inputs',shape=[length,], dtype='int64')
    # x = Embedding(input_dim=21, input_length=length, output_dim=ed, mask_zero=True)(inputs)
    # x = TransformerEncoder(ed, 32, 4)(x)
    # x = GlobalMaxPooling1D()(x)
    # x = Dropout(0.5)(x)
    # outputs = Dense(out_length, activation='sigmoid', name='output', kernel_regularizer=l2(l2value))(x)
    # model = Model(inputs, outputs)
    # return model

    #
    # ed = ['embedding_dimension']
    # ps = ['pool_size']
    # fd = ['fully_dimension']
    # dp = ['drop_out']
    # lr = ['learning_rate']
    # l2value = 0.001

    inputs = Input(name="inputs", shape=[length, length3], dtype="int64")
    # x_1 = Embedding(input_dim=21, input_length=length, output_dim=ed, mask_zero=True)(inputs)

    # x_2 = PositionalEmbedding(sequence_length=length, input_dim=21, output_dim=ed)(inputs)
    a = TransformerEncoder(ed, 32, 4)(inputs)

    a = MaxPooling1D(pool_size=ps, strides=1, padding="same")(a)

    # BiLSTM
    a = Bidirectional(CuDNNLSTM(50, return_sequences=True))(a)
    # LSTM
    # a = CuDNNLSTM(100,return_sequences=True)(a)
    # GRU
    # a = CuDNNGRU(100,return_sequences=True)(a)
    # BiGRU
    # a = Bidirectional(CuDNNGRU(50,return_sequences=True))(a)

    # CNN
    """b = Convolution1D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(l2value))(x_2)
    b = MaxPooling1D(pool_size=ps, strides=1, padding='same')(b)

    merge = Concatenate(axis=-1)([a, b,x_2])"""

    a = MultiHeadAttention(num_heads=2, key_dim=ed)(a, a, a)

    x = GlobalMaxPooling1D()(a)
    x = Dropout(dp)(x)

    x = Dense(fd, activation="relu", name="FC1", kernel_regularizer=l2(l2value))(x)

    outputs = Dense(
        out_length, activation="sigmoid", name="output", kernel_regularizer=l2(l2value)
    )(x)
    model = Model(inputs, outputs)
    adam = Adam(learning_rate=lr)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    return model


def lstm(
    hidden_size,
    length,
    length3,
    out_length,
):
    model = Sequential()

    model.add(LSTM(hidden_size, dropout=0.1, recurrent_dropout=0.1))
    # model.add(Dropout(0.2))
    model.add(Dense(out_length, activation="sigmoid"))
    # model.summary()
    #
    # model = Sequential()
    # model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=( length,length3,)))
    #
    # model.add(LSTM(units=100, activation='relu', return_sequences=False))
    # # model.add(Dropout(0.2))
    # model.add(Dense(2, activation='sigmoid'))

    # model.summary()
    return model


def TextLSTM(
    hidden_size,
    out_length,
):
    model = Sequential()
    # model.add(Embedding(len(tokenizer.index_word) + 1, 10, input_length=max_sequence_length))
    model.add(LSTM(64))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(out_length, activation="softmax"))
    return model


# length 维度
def gru(
    hidden_size,
    out_length,
):

    model = Sequential()
    model.add(GRU(hidden_size, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(out_length, activation="sigmoid"))

    return model


def RNN(
    hidden_size,
    out_length,
):
    model = Sequential()
    model.add(SimpleRNN(hidden_size))
    # model.add(Dropout(0.1))
    model.add(Dense(out_length, activation="sigmoid"))

    return model


def MLP(
    hidden_size,
    out_length,
):
    # model = Sequential()
    #
    # model.add(Flatten())
    # model.add(Dense(hidden_size, activation="relu"))
    # model.add(Dropout(0.2))
    # model.add(Dense(out_length, activation="softmax"))
    model = Sequential()
    model.add(Flatten())
    model.add(
        Dense(
            64,
            activation="relu",
        )
    )
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="sigmoid"))
    return model


def BiLSTM(
    hidden_size,
    out_length,
):
    model = Sequential()
    # model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim))
    # model.add(Bidirectional(LSTM(300)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(hidden_size, activation="relu"))
    # model.add(Dropout(0.1))
    model.add(Dense(out_length, activation="sigmoid"))
    return model


def TextCNN(
    length,
    length3,
    hidden_size,
    out_length,
):
    inputs = Input(name="inputs", shape=[length, length3], dtype="float64")
    layer = inputs

    ## 词窗大小分别为3,4,5
    cnn1 = Conv1D(36, 3, padding="same", strides=1, activation="relu")(layer)
    cnn1 = MaxPooling1D(pool_size=2, data_format="channels_first")(cnn1)
    cnn2 = Conv1D(36, 4, padding="same", strides=1, activation="relu")(layer)
    cnn2 = MaxPooling1D(pool_size=2, data_format="channels_first")(cnn2)
    cnn3 = Conv1D(36, 5, padding="same", strides=1, activation="relu")(layer)
    cnn3 = MaxPooling1D(pool_size=2, data_format="channels_first")(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    x = Flatten()(cnn)
    x = Dense(hidden_size, activation="relu")(x)
    output = Dense(out_length, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=output)
    # model.summary()
    return model


def TextRNN(
    hidden_size,
    out_length,
):
    model = Sequential()
    # model.add(Embedding(len(tokenizer.index_word) + 1, 10, input_length=max_sequence_length))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(Dense(out_length, activation="sigmoid"))
    return model


def CNN(
    hidden_size,
    out_length,
):
    model = Sequential()
    # model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True))
    # model.add(Dropout(0.25))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2, data_format="channels_first"))
    model.add(Flatten())
    # model.add(Dense(hidden_size, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(out_length, activation="softmax"))

    return model


def CNN_LSTM(
    hidden_size,
    out_length,
):
    model = Sequential()
    # model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim))
    # model.add(Dropout(0.25))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2, data_format="channels_first"))
    model.add(LSTM(hidden_size))
    model.add(Dropout(0.2))
    model.add(Dense(out_length, activation="softmax"))
    return model


def Attention(
    length,
    length3,
    hidden_size,
    out_length,
):
    inputs = Input(name="inputs", shape=[length, length3], dtype="float64")
    # x = Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True)(inputs)
    x = MultiHeadAttention(1, key_dim=hidden_size)(inputs, inputs, inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(out_length, activation="softmax")(x)
    model = Model(inputs=[inputs], outputs=output)
    model.summary()

    return model


def MultiHeadAtten(
    length,
    length3,
    hidden_size,
    out_length,
):
    inputs = Input(name="inputs", shape=[length, length3], dtype="float64")
    # x = Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True)(inputs)
    x = MultiHeadAttention(8, key_dim=hidden_size)(inputs, inputs, inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(out_length, activation="softmax")(x)
    model = Model(inputs=[inputs], outputs=output)

    return model


def Attention_BiLSTM(
    length,
    length3,
    hidden_size,
    embed_dim,
    out_length,
):
    inputs = Input(name="inputs", shape=[length, length3], dtype="float64")
    # x = Embedding(top_words, input_length=max_words, output_dim=embed_dim)(inputs)
    x = MultiHeadAttention(1, key_dim=embed_dim)(inputs, inputs, inputs)
    x = Bidirectional(LSTM(hidden_size))(x)
    x = Dense(64, activation="relu")(x)
    # x = Dropout(0.2)(x)
    output = Dense(out_length, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=output)

    return model


def BiGRU_Attention(
    length,
    length3,
    embed_dim,
    out_length,
):
    inputs = Input(name="inputs", shape=[length, length3], dtype="float64")
    # x = Embedding(top_words, input_length=max_words, output_dim=embed_dim)(inputs)
    x = Bidirectional(GRU(32, return_sequences=True))(inputs)
    x = MultiHeadAttention(2, key_dim=embed_dim)(x, x, x)
    x = Bidirectional(GRU(32))(x)
    x = Dropout(0.2)(x)
    output = Dense(out_length, activation="softmax")(x)
    model = Model(inputs=[inputs], outputs=output)
    return model


def seq2seq(length, length3, out_length):
    inputs = Input(shape=(length, length3), dtype="float64")

    x = Bidirectional(GRU(32, return_sequences=True))(inputs)

    # 这里根据需要构建您的多头注意力层
    x = MultiHeadAttention(num_heads=2, key_dim=32)(x, x, x)

    x = Bidirectional(GRU(32))(x)
    x = Dropout(0.2)(x)
    output = Dense(out_length, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=output)
    return model


def ANN(length, out_length):

    # model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(64))
    # model.add(Dense(hidden_size, activation='relu'))
    # model.add(Dense(out_length, activation='sigmoid'))
    # 创建Sequential模型
    model = Sequential()

    model.add(Flatten(input_shape=(1, length)))

    # 添加隐藏层
    model.add(Dense(units=32, activation="relu"))

    # 添加输出层
    model.add(Dense(out_length, activation="sigmoid"))

    # 编译模型

    return model


from torch.autograd import Variable
import torch.nn as nn
import torch


class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        # padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (1, kSize), stride=stride, bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output


class P(nn.Module):
    def __init__(self, kSize, stride=2):
        super().__init__()
        # padding = int((kSize - 1) / 2)
        self.pool = nn.MaxPool2d((1, kSize), stride=stride)

    def forward(self, input):
        output = self.pool(input)
        return output
