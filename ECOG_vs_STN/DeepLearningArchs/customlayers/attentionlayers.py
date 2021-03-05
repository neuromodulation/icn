import keras
import tensorflow as tf
from keras.layers import *
from keras.layers import ReLU
from keras.engine import InputSpec, Layer
from keras import activations, initializers, regularizers, constraints


# USED WTIH Model not Sequential
def self_attention(x, conv=None, k_s=6, atten_func=Dense, d_k=6, activation=ReLU): # NEEDS FIXING
    # input shape should (ch,1), ch is number of ecog channels

    if conv != None:
        input_x = conv(d_k, kernel_size=k_s, padding='same')(x)
    else:
        input_x = x

    query = atten_func(d_k)(input_x)
    query = activation()(query)

    key = atten_func(d_k)(input_x)
    key = activation()(key)

    value = atten_func(d_k, activation='relu')(input_x)
    value = activation()(value)

    matmul = tf.matmul(query, key, transpose_a=True)

    matmul_sqrt = matmul / tf.sqrt(query.shape[1])
    softmax = Softmax()(matmul_sqrt)
    matmul_2 = tf.matmul(value, softmax, transpose_b=False)

    d = Dense(d_k)(matmul_2)
    d = activation()(d)

    d2 = Dense(d_k)(d)
    addition = Add()([x, d2])
    norm = BatchNormalization()(addition)

    return norm



class SpatialAttention(Layer):

    def __init__(self,
                 activation="relu",
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 batch_normalization=False,
                 kernel_size=[1, 128],
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(SpatialAttention, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=4)
        self.supports_masking = True

        def build(self, input_shape):
            time_steps = input_shape[3]
            channels = input_shape[2]
            dimensions = input_shape[1]

            self.attention = keras.models.Sequential(name='attention')

            self.attention.add(keras.layers.Conv2D(dimensions,
                                                   kernel_size=kernel_size,
                                                   input_shape=(
                                                       dimensions, channels, time_steps,),
                                                   kernel_initializer=self.kernel_initializer,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   kernel_constraint=self.kernel_constraint))

            if batch_normalization:
                self.attention.add(keras.layers.BatchNormalization())

            self.attention.add(keras.layers.Activation(self.activation))

            self.attention.add(keras.layers.Conv2D(1,
                                                   kernel_size=kernel_size,
                                                   kernel_initializer=self.kernel_initializer,
                                                   kernel_regularizer=self.kernel_regularizer,
                                                   kernel_constraint=self.kernel_constraint
                                                   ))

            self.attention.add(keras.layers.Activation('sigmoid'))

            self.trainable_weights = self.attention.trainable_weights
            self.non_trainable_weights = self.attention.non_trainable_weights

            self.built = True

            def call(self, inputs):
                attention = self.attention(inputs)

                return keras.layers.Multiply()([inputs, attention])

            def compute_output_shape(self, input_shape):
                return input_shape

            def get_config(self):
                config = {
                    'activation': activations.serialize(self.activation),
                    'kernel_initializer': initializers.serialize(self.kernel_initializer),
                    'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                    'kernel_constraint': constraints.serialize(self.kernel_constraint)
                }

                return dict(config)

# NEEDS Adoptiation to our data shape
class SelfAttention(Layer):
    """
    Implements a self attention mechanism over time series data, weighting
     the input time series by a learned, softmax scaled
      attention matrix.
    # Arguments
        activation: Activation function to use
        if nothing is specified, no activation is applied ('linear')

        kernel_initializer: Initializer for the kernel weights matrix

        kernel_regularizer: Regularizer function applied to
            the kernel weights matrix

        kernel_constraints: Constrain function applied to the kernel
            weights matrix

    # Input shape
            3D tensor with shape: (batch_size, time_step, dimenstion)

    # Output shape
        3D tensor with shape: (batch_size, time_step, scaled_dimensions)

    """

    def __init__(self,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SelfAttention, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=3)
        self.supports_masking = True

        def build(self, input_shape):
            time_steps = input_shape[1]
            dimensions = input_shape[2]

            self.attention = keras.models.Sequential(name='attention')

            self.attention.add(keras.layers.Dense(dimensions,
                                                  input_shape=(
                                                      time_steps, dimensions,),
                                                  kernel_initializer=self.kernel_initializer,
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  kernel_constraint=self.kernel_constraint))

            self.attention.add(keras.layers.Activation(self.activation))

            self.attention.add(keras.layers.Dense(1,
                                                  kernel_initializer=self.kernel_initializer,
                                                  kernel_regularizer=self.kernel_regularizer,
                                                  kernel_constraint=self.kernel_constraint
                                                  ))

            self.attention.add(keras.layers.Flatten())
            self.attention.add(keras.layers.Activation('softmax'))
            self.attention.add(keras.layers.RepeatVector(dimensions))
            self.attention.add(keras.layers.Permute([2, 1]))

            self.trainable_weights = self.attention.trainable_weights
            self.non_trainable_weights = self.attention.non_trainable_weights

            self.built = True

            def call(self, inputs):
                attention = self.attention(inputs)

                return keras.layers.Multiply()([inputs, attention])

            def compute_output_shape(self, input_shape):
                return input_shape

            def get_config(self):
                config = {
                    'activation': activations.serialize(self.activation),
                    'kernel_initializer': initializers.serialize(self.kernel_initializer),
                    'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                    'kernel_constraint': constraints.serialize(self.kernel_constraint)
                }

                return dict(config)


