import numpy as np
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt



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

def plot_prediction(model, x,y, batch_size):
    preds = model.predict(x,batch_size=batch_size, verbose = 1)
    x_ = np.arange(0, preds.shape[0], 1)*0.001
    plt.figure(figsize=(20,5))
    plt.plot(x_, preds, label="val prediction")
    plt.plot(x_, y, label="val label")
    plt.xlabel("Time [s]")
    plt.ylabel("Force")
    plt.title(r"$R^2$"+"="+str(metrics.r2_score(y, preds)))
    plt.legend()
    plt.show()