import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.patches as mpatches
import calendar
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

class AnomalyDetector(object):
    def __init__(self, backward_window_size=30, forward_window_size=14, threshold=5.0, drift=1.0):
        self.backward_window_size = backward_window_size
        self.forward_window_size = forward_window_size
        self.threshold = threshold
        self.drift = drift
        self.anomalies_ = None
        self.anomalies_if_ = None
        self.anomalies_svm_ = None
        self.anomalies_lof_ = None

    def one_pass(self, train_zone, prediction_zone, threshold=None, drift=None):
        if not threshold:
            threshold = self.threshold
        if not drift:
            drift = self.drift

        current_std = np.nanstd(train_zone, ddof=1)
        current_mean = np.nanmean(train_zone)
        drift = drift * current_std
        threshold = threshold * current_std

        x = prediction_zone.astype('float64')
        gp, gn = np.zeros(x.size), np.zeros(x.size)

        for i in range(1, x.size):
            gp[i] = max(gp[i - 1] + x[i] - current_mean - drift, 0)
            gn[i] = min(gn[i - 1] + x[i] - current_mean + drift, 0)

        is_fault = np.logical_or(gp > threshold, gn < -threshold)
        return is_fault

    def detect(self, time_series, threshold=None, drift=None, excluded_points=None):
        if excluded_points is not None:
            time_series[time_series.index.isin(excluded_points)] = np.nan

        ts_values = time_series.values
        ts_index = time_series.index

        detection_series = np.zeros(len(ts_values)).astype('int32')

        for ini_index in range(len(ts_values) - (self.backward_window_size + self.forward_window_size)):
            sep_index = ini_index + self.backward_window_size
            end_index = sep_index + self.forward_window_size
            faults_indexes = self.one_pass(ts_values[ini_index:sep_index],
                                           ts_values[sep_index:end_index],
                                           threshold, drift)
            detection_series[sep_index:end_index][faults_indexes] = 1
        self.anomalies_ = pd.Series(detection_series, index=ts_index)

        return self.anomalies_

    def detect_isolation_forest(self, time_series, contamination=0.05):
        ts_clean = time_series.dropna()
        X = ts_clean.values.reshape(-1, 1)
        X_scaled = (X - X.mean()) / X.std()
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_scaled)
        y_pred = model.predict(X_scaled)
        is_outlier = (y_pred == -1).astype(int)
        self.anomalies_if_ = pd.Series(0, index=time_series.index)
        self.anomalies_if_.loc[ts_clean.index] = is_outlier
        return self.anomalies_if_

    def detect_one_class_svm(self, time_series, nu=0.05, gamma=0.1):
        ts_clean = time_series.dropna()
        X = ts_clean.values.reshape(-1, 1)
        X_scaled = (X - X.mean()) / X.std()
        model = OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
        model.fit(X_scaled)
        y_pred = model.predict(X_scaled)
        is_outlier = (y_pred == -1).astype(int)
        self.anomalies_svm_ = pd.Series(0, index=time_series.index)
        self.anomalies_svm_.loc[ts_clean.index] = is_outlier
        return self.anomalies_svm_

    def detect_lof(self, time_series, n_neighbors=35, contamination=0.05):
        ts_clean = time_series.dropna()
        X = ts_clean.values.reshape(-1, 1)
        X_scaled = (X - X.mean()) / X.std()
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        y_pred = model.fit_predict(X_scaled)
        is_outlier = (y_pred == -1).astype(int)
        self.anomalies_lof_ = pd.Series(0, index=time_series.index)
        self.anomalies_lof_.loc[ts_clean.index] = is_outlier
        return self.anomalies_lof_

    def plot(self, time_series, ax=None, figsize=(14, 7),
             xlabel='Дата', ylabel='тысяч рублей', title='Plot Cusum Anomaly Detection',
             grid=True, marketsize=5):
        anomalies = pd.Series(np.where(self.anomalies_ == 1, time_series, np.nan),
                              index=time_series.index)
        fig, ax = self._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.plot(time_series, label='actual')
        ax.plot(anomalies, 'o', color='r', markersize=marketsize, label='anomalies')
        ax.legend(loc='best')
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    def plot_isolation_forest(self, time_series, ax=None, figsize=(14, 7),
                              xlabel='Дата', ylabel='тысяч рублей', title='Isolation Forest',
                              grid=True, marketsize=5):
        anomalies = pd.Series(np.where(self.anomalies_if_ == 1, time_series, np.nan),
                              index=time_series.index)
        fig, ax = self._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.plot(time_series, label='actual')
        ax.plot(anomalies, 'o', color='r', markersize=marketsize, label='anomalies')
        ax.legend(loc='best')
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    def plot_one_class_svm(self, time_series, ax=None, figsize=(14, 7),
                           xlabel='Дата', ylabel='тысяч рублей', title='One-Class SVM',
                           grid=True, marketsize=5):
        anomalies = pd.Series(np.where(self.anomalies_svm_ == 1, time_series, np.nan),
                              index=time_series.index)
        fig, ax = self._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.plot(time_series, label='actual')
        ax.plot(anomalies, 'o', color='r', markersize=marketsize, label='anomalies')
        ax.legend(loc='best')
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    def plot_lof(self, time_series, ax=None, figsize=(14, 7),
                 xlabel='Дата', ylabel='тысяч рублей', title='Local Outlier Factor',
                 grid=True, marketsize=5):
        anomalies = pd.Series(np.where(self.anomalies_lof_ == 1, time_series, np.nan),
                              index=time_series.index)
        fig, ax = self._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.plot(time_series, label='actual')
        ax.plot(anomalies, 'o', color='r', markersize=marketsize, label='anomalies')
        ax.legend(loc='best')
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

    def hist(self, meas='day', th=0.15, ax=None, figsize=(14, 7),
             xlabel='День месяца', ylabel='количество аномалий', title='Hist Cusum Anomaly Detection',
             grid=True):
        idx, anomaly_count, periodic_anomaly_idx = self.__count_anomaly(th, meas, self.anomalies_)
        simple_color = '#36b2e2'
        anomaly_gradient_colors = dict(zip(periodic_anomaly_idx,
                                           sns.color_palette("Reds", len(periodic_anomaly_idx)).as_hex()[::-1]))
        colors = [simple_color if x[1] / sum(anomaly_count) < th else anomaly_gradient_colors[x[0]]
                  for x in zip(idx, anomaly_count)]
        fig, ax = AnomalyDetector._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.set_xlim(0, max(idx))
        ax.set_ylim(0, max(anomaly_count) + 1)
        ax.bar(idx, anomaly_count, color=colors)

        if meas == 'month':
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])

    
    def hist_isolation_forest(self, meas='day', th=0.15, ax=None, figsize=(14, 7),
                          xlabel='День месяца', ylabel='количество аномалий', title='Isolation Forest',
                          grid=True):
        idx, anomaly_count, periodic_anomaly_idx = self.__count_anomaly(th, meas, self.anomalies_if_)
        simple_color = '#36b2e2'
        anomaly_gradient_colors = dict(zip(periodic_anomaly_idx,
                                        sns.color_palette("Reds", len(periodic_anomaly_idx)).as_hex()[::-1]))
        colors = [simple_color if x[1] / sum(anomaly_count) < th else anomaly_gradient_colors[x[0]]
                for x in zip(idx, anomaly_count)]
        fig, ax = AnomalyDetector._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.set_xlim(0, max(idx))
        ax.set_ylim(0, max(anomaly_count) + 1)
        ax.bar(idx, anomaly_count, color=colors)

        if meas == 'month':
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])


    def hist_one_class_svm(self, meas='day', th=0.15, ax=None, figsize=(14, 7),
                       xlabel='День месяца', ylabel='количество аномалий', title='One-Class SVM',
                       grid=True):
        idx, anomaly_count, periodic_anomaly_idx = self.__count_anomaly(th, meas, self.anomalies_svm_)
        simple_color = '#36b2e2'
        anomaly_gradient_colors = dict(zip(periodic_anomaly_idx,
                                        sns.color_palette("Reds", len(periodic_anomaly_idx)).as_hex()[::-1]))
        colors = [simple_color if x[1] / sum(anomaly_count) < th else anomaly_gradient_colors[x[0]]
                for x in zip(idx, anomaly_count)]
        fig, ax = AnomalyDetector._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.set_xlim(0, max(idx))
        ax.set_ylim(0, max(anomaly_count) + 1)
        ax.bar(idx, anomaly_count, color=colors)

        if meas == 'month':
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])


    def hist_lof(self, meas='day', th=0.15, ax=None, figsize=(14, 7),
             xlabel='День месяца', ylabel='количество аномалий', title='Local Outlier Factor',
             grid=True):
        idx, anomaly_count, periodic_anomaly_idx = self.__count_anomaly(th, meas, self.anomalies_lof_)
        simple_color = '#36b2e2'
        anomaly_gradient_colors = dict(zip(periodic_anomaly_idx,
                                        sns.color_palette("Reds", len(periodic_anomaly_idx)).as_hex()[::-1]))
        colors = [simple_color if x[1] / sum(anomaly_count) < th else anomaly_gradient_colors[x[0]]
                for x in zip(idx, anomaly_count)]
        fig, ax = AnomalyDetector._conf_axs(ax, figsize, xlabel, ylabel, title, grid)
        ax.set_xlim(0, max(idx))
        ax.set_ylim(0, max(anomaly_count) + 1)
        ax.bar(idx, anomaly_count, color=colors)

        if meas == 'month':
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])


    def __count_anomaly(self, th, meas, anomalies):
        anomaly_idx = getattr(anomalies[anomalies == 1].index, meas)
        count_anomalies_by_idx = sorted(Counter(anomaly_idx).items(), key=lambda x: x[1], reverse=True)
        idx = [x[0] for x in count_anomalies_by_idx]
        anomaly_count = [x[1] for x in count_anomalies_by_idx]
        periodic_anomaly_idx = [x[0] for x in count_anomalies_by_idx if x[1] / len(anomaly_idx) >= th]
        return idx, anomaly_count, periodic_anomaly_idx


    @staticmethod
    def _conf_axs(ax, figsize, xlabel, ylabel, title, grid):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True)
        return fig, ax
