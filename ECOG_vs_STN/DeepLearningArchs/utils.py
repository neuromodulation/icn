
import tensorflow as tf


def scheduler(epoch, lr, DECAY_EVERY = 100,DECAY_RATE = 0.95):
    if epoch % DECAY_EVERY == 0:
        return lr * DECAY_RATE
    else:
        return lr


def correlation__(x, y):
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den

