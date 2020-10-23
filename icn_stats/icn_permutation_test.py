import numpy as np
from sklearn import datasets
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

### Write this as a function
def permutationTestMedianSplit(x, y, plot_=True, x_unit=None, p=5000):
    """
    Calculate permutation test
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : distribution that will be compared statistically e.g. R^2
    y (np array) : according ordinal values e.g. UPDRS
    plot_ (boolean) : if True: permutation histplot and ground truth will be potted
    x_unit (str) : histplot xlabel
    p (int): number of permutations
    """

    # first, order y, and take respective values of x
    arr_argsort = x[np.argsort(y)]

    # compute ground truth difference
    gT = np.abs(np.average(arr_argsort[:int(arr_argsort.shape[0]/2)]) -
            np.average(arr_argsort[int(arr_argsort.shape[0]/2):]))

    #
    pV = arr_argsort
    pS = copy.copy(pV)
    #Initialize permutation:
    pD = []
    # Permutation loop:
    for i in range(0,p):
      # Shuffle the data:
        random.shuffle(pS)
        # Compute permuted absolute difference of your two sampled distributions and store it in pD:
        pD.append(np.abs(np.average(pS[0:int(len(pS)/2)]) - np.average(pS[int(len(pS)/2):])))

    # calculate p value
    p_val = len(np.where(pD>=gT)[0])/p

    if plot_ is True:
        plt.hist(pD, bins=30,label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("p="+str(p_val))
        plt.xlabel(r"$R^2$")
        plt.legend()
        plt.show()
    return p_val

    ### Now write a second version where two distributions are compared
    ### Write this as a function
def permutationTest(x, y, plot_=True, x_unit=None, p=5000):
    """
    Calculate permutation test
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distr.
    y (np array) : first distr.
    plot_ (boolean) : if True: permutation histplot and ground truth will be potted
    x_unit (str) : histplot xlabel
    p (int): number of permutations
    """


    # compute ground truth difference
    gT = np.abs(np.average(x) - np.average(y))


    pV = np.concatenate((x,y), axis=0)
    pS = copy.copy(pV)
    #Initialize permutation:
    pD = []
    # Permutation loop:
    for i in range(0,p):
      # Shuffle the data:
        random.shuffle(pS)
        # Compute permuted absolute difference of your two sampled distributions and store it in pD:
        pD.append(np.abs(np.average(pS[0:int(len(pS)/2)]) - np.average(pS[int(len(pS)/2):])))

    # calculate p value
    p_val = len(np.where(pD>=gT)[0])/p

    if plot_ is True:
        plt.hist(pD, bins=30,label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("p="+str(p_val))
        plt.xlabel(r"$R^2$")
        plt.legend()
        plt.show()
    return p_val
