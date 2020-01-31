import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def fitlm(x, y):
    return sm.OLS(y, sm.add_constant(x)).fit()


def fitlm_kfold(x, y, kfold_splits=5):
    model = LinearRegression()
    if isinstance(x, type(np.array([]))) or isinstance(x, type([])):
        x = pd.DataFrame(x)
    if isinstance(y, type(np.array([]))) or isinstance(y, type([])):
        y = pd.DataFrame(y)
    scores, coeffs = [], np.zeros(x.shape[1])
    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
    for i, (train, test) in enumerate(kfold.split(x, y)):
        model.fit(x.iloc[train, :], y.iloc[train, :])
        score = model.score(x.iloc[test, :], y.iloc[test, :])
        # mdl = fitlm(np.squeeze(y.iloc[test,:].transpose()), np.squeeze(model.predict(x.iloc[test, :])))
        scores.append(score)
        coeffs = np.vstack((coeffs, model.coef_))
    coeffs = list(np.delete(coeffs, 0))
    return scores, coeffs, model, ['scores', 'coeffs', 'model']


def zscore(data):
    return (data - data.mean()) / data.std()
