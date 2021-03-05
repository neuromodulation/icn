import keras
from keras.models import Model
from keras.layers import *


def make_aemodel(rnn_model, cnn_model, input_shape=(1, 6, 128), summary=True,
                 output_n=1
                 ):
    input_layer = \
        Input(input_shape)
    #    input_layer_extended = tf.keras.backend.expand_dims(input_layer)

    rnn_model = Model(rnn_model.input, rnn_model.layers[-2].output, name='RNN_Mode')
    cnn_model = Model(cnn_model.input, cnn_model.layers[-2].output, name='CNN_Mode')

    for layer in rnn_model.layers:
        layer.trainable = False

    for layer in cnn_model.layers:
        layer.trainable = False

    rnn_features = rnn_model(input_layer)
    cnn_features = cnn_model(input_layer)

    concat = Concatenate(axis=-1, name='Concat')([rnn_features, cnn_features])

    x = Dense(200, name='Dense_')(concat)
    x = ReLU()(x)
    x = Dense(100, name='Dense2')(x)
    x = ReLU()(x)
    output = Dense(output_n, activation='linear', name='output')(x)

    ae = Model(input_layer, output)

    if summary:
        ae.summary()

    return ae
