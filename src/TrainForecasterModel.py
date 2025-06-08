import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from ChangeDetection import  *
from FeatureExtractor import *
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

class TrainForecaster:
    def __init__(self, time_series, index_start, lags, calendar_extractor, n_preds=7, selected_features=None, use_feature_selection=True):
        self.time_series = time_series.copy()
        self.index_start = index_start
        self.lags = lags
        self.calendar_extractor = calendar_extractor
        self.n_preds = n_preds
        self.selected_features = selected_features
        self.use_feature_selection = use_feature_selection
        
        self.known_drift_points = []
        self.drift_dates = []
        self.df_train = None
        self.weekday_mean = None
        self.month_mean = None

        self.calendar_features = ['is_weekends', 'is_preholiday', 'day_of_month',
       'day_of_year', 'week_of_year', 'is_month_start', 'is_month_end', 'wd_0',
       'wd_1', 'wd_2', 'wd_3', 'wd_4', 'wd_5', 'wd_6', 'm_1', 'm_2', 'm_3',
       'm_4', 'm_5', 'm_6', 'm_7', 'm_8', 'm_9', 'm_10', 'm_11', 'm_12', 'q_1',
       'q_2', 'q_3', 'q_4', 's_autumn', 's_spring', 's_summer', 's_winter',
       'is_25','is_26','is_9','is_10','is_5','is_27', 'is_tax_week']
        
        self.predictions_dict = {}
        
        self.new_dataset = None
        self.update_dataset = None
        self.period_kalib = []
        self.period_balance = []

        self._initialize()

    def _initialize(self):
        """Инициализация обучающего набора данных и первой модели."""
        self.X_train, self.y_train, self.drift_dates, self.known_drift_points, self.df_train, self.weekday_mean, self.month_mean = self._initialize_training_data()
        self.lag_features = list(set(list(self.X_train.columns)) - set(self.calendar_features))
        self.pipeline = self._create_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

        future_forecast = self._update_forecast()
        future_dates = pd.date_range(start=self.df_train.index[-1] + pd.Timedelta(days=1), periods=self.n_preds, freq='D')
        self._update_predictions(future_forecast, future_dates)

        self.new_dataset = self.time_series[:self.index_start].copy()
        self.update_dataset = self.time_series[:self.index_start].copy()

    def _initialize_training_data(self):
        """Подготовка первого обучающего набора."""
        known_drift_points = change_detection(np.array(self.time_series[:self.index_start]))
        drift_dates = [self.time_series[:self.index_start].index[i] for i in known_drift_points]
        df_train = self.calendar_extractor.enrich_with_calendar_features(self.time_series[:self.index_start])
        df_train['regime'] = assign_regime(self.time_series[:self.index_start].index, drift_dates)
        df_train = pd.get_dummies(df_train, columns=['regime'], prefix='reg')
        X_train, y_train, weekday_mean, month_mean = train_dataset(df_train, self.lags)

        if self.selected_features is not None:
          X_train = X_train[self.selected_features]

        return X_train, y_train, drift_dates, known_drift_points, df_train, weekday_mean, month_mean

    
    def _create_pipeline(self):
        """Создание пайплайна с раздельным отбором фич."""

        if not self.use_feature_selection:
            return Pipeline([
                ('regression', LinearRegression())
            ])

        lag_selector = Pipeline([
            ('select_lags', SelectKBest(score_func=f_regression, k=10))  # Например, отбираем 10 лучших лагов
        ])

        calendar_selector = Pipeline([
            ('select_calendar', SelectKBest(score_func=f_regression, k='all'))  # Например, 5 лучших календарных фичей
        ])

        # Разделяем по колонкам
        feature_union = ColumnTransformer([
            ('lags', lag_selector, self.lag_features),             # список колонок с лагами
            ('calendar', calendar_selector, self.calendar_features) # список колонок с календарными фичами
        ])

        pipeline = Pipeline([
            ('feature_processing', feature_union),
            ('regression', LinearRegression())
        ])
    
        return pipeline

    def _update_forecast(self):
        """Создание прогноза на будущее."""
        forecast = forecast_method(
            self.df_train,
            self.lags,
            self.pipeline,
            self.weekday_mean,
            self.month_mean,
            self.X_train.columns,
            self.n_preds,
            self.drift_dates,
            self.calendar_extractor
        )
        return forecast['predicted_Balance']

    def _retrain_model(self, dataset):
        """Переобучение модели при разладке."""
        self.drift_dates = [dataset.index[i] for i in self.known_drift_points]
        self.df_train = self.calendar_extractor.enrich_with_calendar_features(dataset)
        self.df_train['regime'] = assign_regime(dataset.index, self.drift_dates)
        self.df_train = pd.get_dummies(self.df_train, columns=['regime'], prefix='reg')
        self.X_train, self.y_train, self.weekday_mean, self.month_mean = train_dataset(self.df_train, self.lags)

        if self.selected_features is not None:
          self.X_train = self.X_train[self.selected_features]
        
        self.lag_features = list(set(list(self.X_train.columns)) - set(self.calendar_features))
        self.pipeline = self._create_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def _update_predictions(self, future_forecast, future_dates):
        """Обновление словаря прогнозов."""
        for date, value in zip(future_dates, future_forecast):
            self.predictions_dict[pd.to_datetime(date)] = float(value)

    def run(self):
        """Основной цикл обработки новых данных."""
        for new_data in self.time_series[self.index_start:].index:
            new_real = self.time_series.loc[new_data, 'Balance']
            self.period_kalib.append(new_real)
            self.period_balance.append((new_data, new_real))

            new_row = pd.DataFrame({'Balance': [new_real]}, index=[new_data])
            self.update_dataset = pd.concat([self.update_dataset, new_row])

            drift_found, new_points = check_new_drift(
                np.array(self.update_dataset['Balance']),
                old_detected=self.known_drift_points,
                tolerance=15,
                window_size=42
            )

            if drift_found and new_points and (new_points[0] - self.known_drift_points[-1]) <= 90:
                drift_found = False

            if drift_found:
                if len(new_points) > 1:
                    new_points = [new_points[0]]

                print(f"⚠ Новая разладка: {new_data}, реально {self.update_dataset.index[new_points]}")

                self.known_drift_points += new_points
                print(f'Ошибки за {len(self.period_balance)} недель:', count_large_errors_period(self.predictions_dict, self.period_balance))

                self.period_kalib, self.period_balance = [], []
                self.new_dataset = self.update_dataset.copy()

                self._retrain_model(self.update_dataset)

                future_forecast = self._update_forecast()
                future_dates = pd.date_range(start=self.df_train.index[-1] + pd.Timedelta(days=1), periods=self.n_preds, freq='D')
                self._update_predictions(future_forecast, future_dates)

            elif len(self.period_kalib) == self.n_preds:
                print(f'Ошибки за {len(self.period_balance)} недель:', count_large_errors_period(self.predictions_dict, self.period_balance))

                dates, values = zip(*self.period_balance)
                week_series = pd.DataFrame({'Balance': values}, index=dates)
                self.new_dataset = pd.concat([self.new_dataset, week_series])

                self.period_kalib, self.period_balance = [], []
                self._retrain_model(self.new_dataset)

                future_forecast = self._update_forecast()
                future_dates = pd.date_range(start=self.df_train.index[-1] + pd.Timedelta(days=1), periods=self.n_preds, freq='D')
                self._update_predictions(future_forecast, future_dates)

                print(f'Баланс на {new_data + pd.Timedelta(days=1)}: {self.predictions_dict[new_data + pd.Timedelta(days=1)]}')
            else:
                print(f'Баланс на {new_data + pd.Timedelta(days=1)}: {self.predictions_dict[new_data + pd.Timedelta(days=1)]}')

    def get_history_log(self):
        """Сформировать историю реальных и предсказанных значений."""
        history_log = []
        for pred_date in self.predictions_dict.keys():
            if pred_date in self.time_series.index:
                real_value = self.time_series.loc[pred_date, 'Balance']
            else:
                real_value = 0
            pred_value = self.predictions_dict[pred_date]
            history_log.append((pred_date, real_value, pred_value))
        return history_log
