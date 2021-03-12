import IO
import numpy as np
import label_generator
from scipy import signal
from sklearn.model_selection import train_test_split

def get_f_(sub, vhdr_files, sess = 'right',loc = 'ECOG' ):

    return [file for file in vhdr_files if sub in file and sess in file]

def get_splits(sub, sess, loc, f_, PATH, fs_new = 128, fs = 1000):
    X, y_con, y_ips = IO.get_data_raw_combined(sub, sess, loc, f_,PATH)  # Raw Data

    # resample data to fs_new
    y_con = signal.resample(y_con, int(y_con.shape[0 ] *fs_new / fs), axis=0)
    y_ips = signal.resample(y_ips, int(y_ips.shape[0 ] *fs_new / fs), axis=0)
    X = signal.resample(X, int(X.shape[1 ] *fs_new / fs), axis=1).T
    X_here = X

    for lat in ["CON", "IPS"]:
        if lat == "CON":
            y_ = y_con
            y__ = y_ips
        else:
            y_ = y_ips
            y__ = y_con

    print("RUNNING subject  " +str(sub ) +" sess:  " +str(sess ) +" lat:  "+ str(lat )+ "loc:  " +str(loc))

    X_train, X_test, y_train, y_test, y__train, y__test = train_test_split(X_here, y_, y__, train_size=0.7 ,shuffle=False)

    X_train, X_val, y_train, y_val, y__train, y__val = train_test_split(X_train, y_train, y__train, train_size=0.7 ,shuffle=False)


    return X_train, y_train, y__train, X_val, y_val, y__val, X_test, y_test, y__test


def tabnet_singlechanneldata(X, y, channel_indx, batch_size, samples, rebalance=True):
    REBALANCE_THR = 0.1
    gen = label_generator.generator_regression(X[:, [channel_indx]], y, batch_size,
                                               samples, rebalance=rebalance, rebalanced_thr=REBALANCE_THR)
    steps = int(X.shape[0] / batch_size)

    result_X = []
    result_y = []
    for step in range(steps):
        batchX, batchY = next(gen)
        result_X.append(batchX)
        result_y.append(batchY)

    resultx = np.array(result_X).reshape(steps * batch_size, samples * channel_indx)
    resulty = np.array(result_y).reshape(steps * batch_size, 1)

    print("steps number =", steps)
    print("Finished ")

    return resultx, resulty


def tabnet_multichanneldata(X, y, batch_size, samples, rebalance=True):
    REBALANCE_THR = 0.1

    ch_n = X.shape[1]
    steps = int(X.shape[0] / batch_size)

    result_y = []
    total_x = np.zeros((steps * batch_size, samples * ch_n))
    for ch in range(ch_n):
        # print(ch)
        gen = label_generator.generator_regression(X[:, [ch]], y, batch_size, \
                                                   samples, rebalance=rebalance, rebalanced_thr=REBALANCE_THR)

        result_X = []

        for step in range(steps):
            batchX, batchY = next(gen)
            result_X.append(batchX)
            if ch == 0:
                result_y.append(batchY)

        total_x[:, samples * ch:samples * (ch + 1)] = np.array(result_X).reshape(steps * batch_size, samples)
    resulty = np.array(result_y).reshape(steps * batch_size, 1)

    print("steps number =", steps)
    print("Finished ")

    return total_x, resulty

