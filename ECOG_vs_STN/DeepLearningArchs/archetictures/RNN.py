import keras
from keras.models import Sequential
from keras.layers import *



def make_rnnModel(input_shape=(1, 6, 128), summary=True
                  , n_dense=3, dense_units=[100, 100, 200], dense_activation=ReLU
                  , n_lstm=2, lstm_units=[128, 128], lstm_activation=None
                  , output_layer=Dense, output_units=1, output_activation=None,
                  ch_first=False):
    if ch_first:
        keras.backend.set_image_data_format('channels_first')

    if ch_first == False:
        keras.backend.set_image_data_format('channels_last')

    try:
        len(dense_units) == n_dense

    except ValueError:
        print('the length of the dens_unit list should be the same as n_dense')

    #    input_layer = keras.backend.squeeze(Input(input_shape), axis =1)

    # starting the model
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Lambda(keras.backend.squeeze, arguments={"axis": 1}))

    # masking the dense part of the network
    for i in range(n_dense):
        model.add(Dense(dense_units[i]))
        model.add(dense_activation())

    for i in range(n_lstm):

        if i + 1 == n_lstm:
            model.add(LSTM(lstm_units[0], return_sequences=False))
        else:
            model.add(LSTM(lstm_units[0], return_sequences=True))

        if lstm_activation:
            model.add(lstm_activation())

    if output_activation == None:
        output_activation = 'linear'

    model.add(output_layer(output_units, activation=output_activation))

    if summary:
        model.summary()

    return model