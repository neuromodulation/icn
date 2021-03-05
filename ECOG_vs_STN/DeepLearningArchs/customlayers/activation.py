import tensorflow as tf
import tensorflow.keras as keras



class SLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SLU, self).__init__(**kwargs)


    def build(self, input_shape):
        pass

    def call(self, x):

        return tf.keras.activations.swish(x)


from tensorflow.keras import layers


class Sinosoidal(layers.Layer):
    def __init__(self, **kwargs):
        super(Sinosoidal, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):

        return tf.math.sin(inputs)


class WeightedSinosoidal(layers.Layer):
    def __init__(self, units, initializer="he_normal", **kwargs):
        super(WeightedSinosoidal, self).__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)
        self.units = units

        if keras.backend.image_data_format() == 'channels_last':
            self.ch_indx = -1
            self.transpose = True
        else:
            self.ch_indx = 1
            self.transpose = False

    def build(self, input_shape):
        output_dim = input_shape[self.ch_indx]
        self.alpha = self.add_weight(
            #            shape=input_shape[1:],
            shape=(1, output_dim, self.units),
            initializer=self.initializer,
            name="alpha",
            trainable=True)
        self.beta = self.add_weight(
            shape=input_shape[1:],
            initializer=self.initializer,
            name="beta",
            trainable=True)

    def call(self, inputs):

        #        mixed = tf.matmul(tf.math.sin(inputs + self.beta),self.alpha, transpose_a=self.transpose)
        result = tf.math.sin(inputs + self.beta)
        result = tf.keras.layers.Dot(axes=(1, 1))([self.alpha, result])
        #        result = Dense(self.units)(tf.math.sin(inputs + self.beta))

        return result

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(WeightedSinosoidal, self).get_config()
        config = {"initializer": keras.initializers.serialize(self.initializer)}
        return dict(list(base_config.items()) + list(config.items()))