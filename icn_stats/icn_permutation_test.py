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
def permutationTestSpearmansRho(x, y, plot_=True, x_unit=None, p=5000):
    """
    Calculate permutation test for multiple repetitions of Spearmans Rho
    https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d

    x (np array) : first distibution e.g. R^2
    y (np array) : second distribution e.g. UPDRS
    plot_ (boolean) : if True: permutation histplot and ground truth will be potted
    x_unit (str) : histplot xlabel
    p (int): number of permutations

    returns:
    gT (float) : estimated ground truth, here spearman's rho
    p (float) : p value of permutation test
    """

    # compute ground truth difference
    gT = stats.spearmanr(x, y)[0]
    #
    pV = np.array((x,y))
    #Initialize permutation:
    pD = []
    # Permutation loop:
    args_order = np.arange(0,pV.shape[1],1)
    args_order_2 = np.arange(0,pV.shape[1],1)
    for i in range(0,p):
      # Shuffle the data:
        random.shuffle(args_order)
        random.shuffle(args_order_2)
        # Compute permuted absolute difference of your two sampled distributions and store it in pD:
        pD.append(stats.spearmanr(pV[0,args_order], pV[1,args_order_2])[0])

    # calculate p value
    if gT < 0:
        p_val = len(np.where(pD<=gT)[0])/p
    else:
        p_val = len(np.where(pD>=gT)[0])/p

    if plot_ is True:
        plt.hist(pD, bins=30,label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth "+x_unit+"="+str(gT)+" p="+str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()
    return gT, p_val

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

    returns:
    gT (float) : estimated ground truth, here abs difference of distribution means
    p (float) : p value of permutation test

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
    if gT < 0:
        p_val = len(np.where(pD<=gT)[0])/p
    else:
        p_val = len(np.where(pD>=gT)[0])/p

    if plot_ is True:
        plt.hist(pD, bins=30,label="permutation results")
        plt.axvline(gT, color="orange", label="ground truth")
        plt.title("ground truth "+x_unit+"="+str(gT)+" p="+str(p_val))
        plt.xlabel(x_unit)
        plt.legend()
        plt.show()
    return gT, p_val
