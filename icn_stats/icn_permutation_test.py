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

def permutationTest_relative(x, y, plot_=True, x_unit=None, p=5000):
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
    gT = np.abs(np.average(x) - np.average(y))
    pD = []
    for i in range(0,p):
        l_ = []
        for i in range(x.shape[0]):
            if random.randint(0,1) == 1:
                l_.append((x[i], y[i]))
            else:
                l_.append((y[i], x[i]))
        pD.append(np.abs(np.average(np.array(l_)[:,0])- np.average(np.array(l_)[:,1])))
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

def cluster_wise_p_val_correction(p_arr, p_sig=0.05, num_permutations=100):
    """

    Based on: https://github.com/neuromodulation/wjn_toolbox/blob/4745557040ad26f3b8498ca5d0c5d5dece2d3ba1/mypcluster.m
    https://garstats.wordpress.com/2018/09/06/cluster/

    Obtain cluster-wise corrected p values.

    p_arr (np.array) : ndim, can be time series or image
    p_sig (float) : significance level
    num_permutations (int) : number of random permutations of cluster comparisons

    returns:
    p (float) : significance level of highest cluster
    p_min_index : indices of significant samples
    """
    labels, num_clusters = measure.label(p_arr<p_sig, return_num=True)

    # loop through clusters of p_val series or image
    index_cluster = {}
    p_cluster_sum = np.zeros(num_clusters)
    for cluster_i in range(num_clusters):
        index_cluster[cluster_i] = np.where(labels == cluster_i+1)[0] # first cluster is assigned to be 1 from measure.label
        p_cluster_sum[cluster_i] = np.sum(np.array(1-p_arr)[index_cluster[cluster_i]])
    p_min = np.max(p_cluster_sum) # p_min corresponds to the most unlikely cluster
    p_min_index = index_cluster[np.argmax(p_cluster_sum)]

    # loop through random permutation cycles
    r_per_arr = np.zeros(num_permutations)
    for r in range(num_permutations):
        r_per = np.random.randint(low=0, high=p_arr.shape[0], size=p_arr.shape[0])

        labels, num_clusters = measure.label(p_arr[r_per]<p_sig, return_num=True)

        index_cluster = {}
        p_cluster_sum = np.zeros(num_clusters)
        for cluster_i in range(num_clusters):
            index_cluster[cluster_i] = np.where(labels == cluster_i+1)[0] # first cluster is assigned to be 1 from measure.label
            p_cluster_sum[cluster_i] = np.sum(np.array(1-p_arr[r_per])[index_cluster[cluster_i]])
        r_per_arr[r] = np.max(p_cluster_sum) # corresponds to the most unlikely cluster

        sorted_r =  np.sort(r_per_arr)

    def find_arg_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    p = 1 - find_arg_nearest(sorted_r, p_min) / num_permutations

    return p, p_min_index
