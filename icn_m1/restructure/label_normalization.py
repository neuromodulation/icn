import filter
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import cvxpy as cp
from scipy import signal


def NormalizeData(data):
    """Aux function of baseline_correction."""
    minv = np.min(data)
    maxv = np.max(data)
    data_new = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_new, minv, maxv


def DeNormalizeData(data, minv, maxv):
    """Aux function of baseline_correction."""
    data_new = (data + minv) * (maxv - minv)
    return data_new


def baseline_als(y, lam, p, niter=10):
    """
    Baseline drift correction based on [1].

    A linear problem is solved: (W + lam*D'*D)z=Wy, where W=diag(w) and
    D=second order diff. matrix.

    Parameters
    ----------
    y : array
        raw signal to be cleaned
    lam : float
        reg. parameter. lam > 0
    p : int
        asymmetric parameter. Value in (0 1).
    niter : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    z : array
        the basesline to be subtracted.

    References
    ----------
    [1] P. H. C. Eilers, H. F. M. Boelens, Baseline correction with asymmetric
    least squares smoothing, Leiden University Medical Centre report, 2005.
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def baseline_rope(y, lam=1):
    """
    Baseline drift correction based on [1].

    Problem to Solve min |y-b| + lam*(diff_b)^2, s.t. b<=y.

    Parameters
    ----------
    y : array
        raw signal to be cleaned..
    lam : float (Default is 1)
        reg. parameter. lam > 0

    Returns
    -------
    z : array
        the basesline to be subtracted.

    References
    ----------
    [1] Xie, Z., Schwartz, O., & Prasad, A. (2018). Decoding of finger
    trajectory from ECoG using deep learning. Journal of neural engineering,
    15(3), 036009.
    """
    b = cp.Variable(y.shape)
    objective = cp.Minimize(cp.norm(y-b, 2)+lam*cp.sum_squares(cp.diff(b, 1)))
    constraints = [b <= y]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver="SCS")
    z = b.value

    return z


def baseline_correction(y, method='baseline_rope', param=1e4, thr=2e-1,
                        normalize=True, Decimate=1, Verbose=True):
    """
    Baseline correction is applied to the label.

    Parameters
    ----------
    y : array/np.array
        raw signal to be corrected
    method : string, optional
        two possible method for baseline correction are allowed 'baseline_rope'
        and 'baseline_als'. See documentation of each method. The default is
        'baseline_rope'.
    param : number or array of numbers, optional
        parameters needed in each optimization method. If baseline_rope is
        being used, "param" refers to the regularization parameter.
        If baseline_als is being used  "param" should be a 2-lenght array where
        the first value is the regularization parameter and the second is the
        weigthed value. The default is [1e2, 1e-4].
    thr : number, optional
        threshold value in each small variation between trails could still
        remains after baseline elimination. The default is 1e-1.
    normalize : boolean, optional
        if normalize is True the original signal as well as the output
        corrected signal will be scalled between 0 and 1. The default is True.
    Decimate: number, optinal
        before baseline correction it might be necessary to downsample the
        original raw signal. We recommend to do this step when long processing
        times are willing to be avoided. The default is 1, i.e. no decimation.
    Verbose: boolean, optional
        The default is True.

    Returns
    -------
    y_corrected: signal with baseline correction
    onoff: squared signal useful for onset target evaluation.
    y: original signal
    """
    if Decimate != 1:
        if Verbose:
            print('>>Signal decimation is being done')
        y = signal.decimate(y, Decimate)

    if method == 'baseline_als' and np.size(param) != 2:
        raise ValueError("If baseline_als method is desired, param should be a"
                         "2 length object")
    if method == 'baseline_rope' and np.size(param) > 1:
        raise ValueError("If baseline_rope method is desired, param should be"
                         " a number")

    if method == 'baseline_als':
        if Verbose:
            print('>>baseline_als is being used')
        z = baseline_als(y, lam=param[0], p=param[1])
    else:
        if Verbose:
            print('>>baseline_rope is being used')
        z = baseline_rope(y, lam=param)

    # subtract baseline
    y_corrected = y-z

    # aux step: normalize to eliminate interferation
    y_corrected, minv, maxv = NormalizeData(y_corrected)

    # eliminate interferation
    y_corrected[y_corrected < thr] = 0
    # create on-off signal
    onoff = np.zeros(np.size(y_corrected))
    onoff[y_corrected > 0] = 1

    if normalize:
        y, Nan, Nan = NormalizeData(y)
    else:
        y_corrected = DeNormalizeData(y_corrected, minv, maxv)
    return y_corrected, onoff, y
