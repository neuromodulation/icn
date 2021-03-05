import keras
from keras.models import Sequential
from keras.layers import *

def make_cnnModel(input_shape=(1, 6, 128), summary=True
                  , n_conv=3, conv_units=[32, 64, 128]
                  , kernel_sizes=[[64, 1], [64, 1], [64, 1]], conv_activation=ReLU, conv=Conv2D
                  , pool_func=MaxPool2D, pool=[0, 1, 1]
                  , n_dense=2, dense_units=[200, 120], dense_activation=ReLU
                  , output_layer=Dense, output_units=2, output_activation=None
                  , ch_first=True
                  , batch_norm=False):
    if ch_first:
        keras.backend.set_image_data_format('channels_first')

    model = Sequential()
    model.add(Input(input_shape))
    #     model.add(Permute((2,1,3) ))

    for i in range(n_conv):
        model.add(conv(conv_units[i], kernel_sizes[i], padding='same'))
        model.add(conv_activation())
        if batch_norm:
            model.add(BatchNormalization())

        #         model.add(DepthwiseConv2D(conv_units[i], kernel_sizes[i]))
        if pool[i] == 1:
            model.add(pool_func(pool_size=(2), strides=(2), padding='same'))

    #     for i in range(n_conv):
    #         model.add(DepthwiseConv2D(conv_units[i], kernel_sizes[i]))
    #         model.add(conv_activation())

    model.add(Flatten())

    for i in range(n_dense):
        model.add(Dense(dense_units[i]))
        model.add(dense_activation())

    model.add(output_layer(output_units, activation='sigmoid'))

    if summary:
        model.summary()

    return model
