from sklearn.model_selection import GroupShuffleSplit
from sklearn import utils, metrics
import numpy as np
from matplotlib import pyplot as plt

def balanced_leave_k_groups_out_CV(model, X, y, plt_=True, same_size=True):

    """

    Balanced 10 fold CV based on group segments, which are resembled of individual movements
    Same label size (movement, rest) can be enabled

    params
    model: sklearn model,
    X (np array) features,
    y (np array) labels,
    plt_ (boolean) enable plotting movement flanks, groups and predictions,
    same_size (boolean) enable resampling of low thresholds <0.01 labels equally to high threshold >0.1 labels

    returns
    r2 (list) : test set prediction R^2 results,
    y_test_pr (list) : test set predictions,
    y_test (list) : test labels
    
    """

    crossings = np.where(np.diff(np.array(y>0.1).astype(int))>0)[0]

    if plt_ is True:
        plt.figure(figsize=(12,5))
        plt.plot(y)
        crossings = np.where(np.diff(np.array(y>0.1).astype(int))>0)[0]
        plt.plot(crossings, y[crossings], "x", color="red")
        plt.title("crossings")

    group_id = 0
    groups = np.zeros([y.shape[0]])
    for i in range(0,y.shape[0]):
        if i in crossings:
            group_id +=1
        groups[i] = group_id

    if plt_ is True:
        plt.figure(figsize=(12,5))
        plt.plot(groups/np.unique(groups).shape[0], label="group")
        plt.plot(y)
        plt.legend()
        plt.title("group identification")
        plt.show()

    gss = GroupShuffleSplit(n_splits=10, train_size=.9, random_state=42)
    r2_res = []
    y_test_pr_ = []
    y_test_ = []

    for train, test in gss.split(X, y, groups):
        X_train = X[train,:]; X_test = X[test,:]
        y_train = y[train]; y_test = y[test]

        #optionally sample balanced from train set
        y_high = y_train[np.where(np.array(y_train>0.1)>0)[0]]
        X_tr_high = X_train[np.where(np.array(y_train>0.1)>0)[0]]

        y_low = y_train[np.where(np.array(y_train<0.01)>0)[0]]
        X_tr_low = X_train[np.where(np.array(y_train<0.01)>0)[0]]

        # shuffle data, then optionally clip to ensure same shape as movement
        X_tr_low, y_low = utils.shuffle(X_tr_low, y_low)

        if same_size is True:
            X_tr_low = X_tr_low[:y_high.shape[0]]
            y_low = y_low[:y_high.shape[0]]

        # concatenate low and high segments
        X_train = np.concatenate((X_tr_low, X_tr_high))
        y_train = np.concatenate((y_low, y_high))
        X_train, y_train = utils.shuffle(X_train, y_train)

        model.fit(X_train, y_train)
        y_test_pr = model.predict(X_test)
        r2_here = metrics.r2_score(y_test, y_test_pr)

        if plt_ is True:
            plt.figure(figsize=(12,5))
            plt.plot(y_test, label="label")
            plt.plot(y_test_pr, label="prediction")
            plt.legend()
            plt.title("r2="+str(np.round(r2_here,2)))
            plt.show()

        if r2_here <0:
            r2_here = 0
        r2_res.append(r2_here)
        y_test_pr_.append(y_test_pr); y_test_.append(y_test)

    print("overall mean: "+str(np.round(np.mean(r2_res),2)))

    return r2_res, y_test_pr_, y_test_
