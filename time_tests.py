import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn import linear_model
import statsmodels.api as sm
from time import time

class TimeTests:
    def __init__(self, cols=5, multiplier=100, logit=False):
        self.cols = cols
        self.multiplier = multiplier
        self.logit = logit

    def _synthetic_data(self, n):
        X = self.multiplier * np.random.random_sample((n, self.cols))
        y = self.multiplier * np.random.random_sample((n, 1))
        X_pred = self.multiplier * np.random.random_sample((int(n * 0.2), self.cols))

        if self.logit == True:
            y = np.round(y / self.multiplier, 0)

        return X, y, X_pred

    def _time_tests(self, sklearn_model, statsmodels_model):
        data = {10 ** n: self._synthetic_data(n=10 ** n) for n in range(2, 9)}
        self.sklearn_times = {}
        for n in data.keys():
            times = []
            for _ in range(10):
                start = time()
                sklearn_lr = sklearn_model(fit_intercept=True)
                sklearn_lr.fit(data[n][0], data[n][1].ravel())
                preds = sklearn_lr.predict(data[n][2])
                times.append(time() - start)
            self.sklearn_times[n] = times

        self.statsmodels_times = {}
        for n in data.keys():
            times = []
            for _ in range(10):
                start = time()
                X_constant = sm.add_constant(data[n][0])
                X_pred_constant = sm.add_constant(data[n][2])
                if statsmodels_model == sm.OLS:
                    statsmodels_lr = statsmodels_model(data[n][1], X_constant)
                    statsmodels_results = statsmodels_lr.fit()
                elif statsmodels_model == sm.Logit:
                    statsmodels_lr = statsmodels_model(data[n][1], X_constant)
                    statsmodels_results = statsmodels_lr.fit(solver="lbfgs")

                preds = statsmodels_lr.predict(statsmodels_results.params,
                                                X_pred_constant)
                times.append(time() - start)

            self.statsmodels_times[n] = times

    def _ttests(self):
        self.ttest_results = {}
        self.means = {}
        for n in self.sklearn_times.keys():
            sklearn = self.sklearn_times[n]
            statsmodels = self.statsmodels_times[n]
            ttest_results = ttest_ind(sklearn, statsmodels, equal_var=False)
            self.ttest_results[n] = ttest_results
            self.means[n] = {"sklearn": np.round(np.mean(sklearn), 4),
                             "statsmodels": np.round(np.mean(statsmodels), 4)}

    def _corrected_significance(self, length):
        self.corrected_significance = 0.05 / length

    def execute(self):
        if self.logit == True:
            self._time_tests(sklearn_model=linear_model.LogisticRegression,
                             statsmodels_model=sm.Logit)

        elif self.logit == False:
            self._time_tests(sklearn_model=linear_model.LinearRegression,
                             statsmodels_model=sm.OLS)

        self._ttests()
        self._corrected_significance(length=len(self.ttest_results))

        self.results = pd.DataFrame(columns=["Number_of_Rows", "sklearn_mean_time",
                                             "statsmodels_mean_time", "t-statistic",
                                             "p-value", "Reject_Null"])

        for n in self.ttest_results.keys():
            reject_null = True if self.ttest_results[n].pvalue < self.corrected_significance else False
            self.results = self.results.append({"Number_of_Rows": n,
                                                "sklearn_mean_time": self.means[n]["sklearn"],
                                                "statsmodels_mean_time": self.means[n]["statsmodels"],
                                                "t-statistic": np.round(self.ttest_results[n].statistic, 4),
                                                "p-value": np.round(self.ttest_results[n].pvalue, 4),
                                                "Reject_Null": reject_null},
                                                ignore_index=True)











