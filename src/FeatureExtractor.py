import pandas as pd
import numpy as np
from CalendarExtractor import CalendarFeatureExtractor

def assign_regime(index, drift_points):
    drift_points = sorted(drift_points)
    return index.map(lambda date: sum(date >= dp for dp in drift_points))

def get_regime_for_date(date, drift_dates):
    return sum(date >= d for d in drift_dates)

def count_large_errors(actual, predicted, threshold=0.42):
    """
    Считает, сколько раз абсолютная ошибка прогноза превышает заданный порог.

    Parameters:
        actual (array-like): Фактические значения.
        predicted (array-like): Прогнозируемые значения.
        threshold (float): Порог ошибки.

    Returns:
        int: Количество превышений порога.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    absolute_errors = np.abs(predicted - actual)
    count_exceeding = np.sum(absolute_errors > threshold)
    return count_exceeding

def count_large_errors_period(predictions_dict, period_balance, threshold=0.42):
    """
    Подсчитывает количество случаев, когда абсолютная ошибка прогноза превышает заданный порог за определенный период времени.

    Parameters:
    predictions_dict : dict
        Словарь с прогнозными значениями, где ключ — это дата,а значение — прогноз.
    
    period_balance : период, за который шел прогноз
    
    threshold (float): Порог ошибки.

    Returns:
        int: Количество превышений порога.
    """
    count = 0
    for date, real_value in period_balance:
        pred_value = predictions_dict[date]
        if abs(pred_value - real_value) > threshold:
            count += 1
    return count

def train_dataset(data, lags):
    train_data = pd.DataFrame(data.copy())

    # Создаем лаги только для обучающей выборки
    for ll in lags:
        train_data[f'lag_{ll}'] = train_data['Balance'].shift(ll)

    for wind in [30,42,90]:
        train_data[f'sma_{wind}'] = train_data['Balance'].rolling(window=wind).mean()
        train_data[f'std_{wind}'] = train_data['Balance'].rolling(window=wind).std()
    #     train_data[f'min_{wind}'] = train_data['Balance'].rolling(window=wind).min()
    #     train_data[f'max_{wind}'] = train_data['Balance'].rolling(window=wind).max()

    period = 66.7
    t = np.arange(len(train_data))  # Создаем последовательность чисел от 0 до длины train_data - 1
    train_data['sin_mid_term'] = np.sin(2 * np.pi * t / period)
    train_data['cos_mid_term'] = np.cos(2 * np.pi * t / period)

    # Добавляем категориальные признаки
    train_data['weekday'] = train_data.index.weekday
    train_data['month'] = train_data.index.month
    weekday_mean = train_data.groupby('weekday')['Balance'].mean().to_dict()
    month_mean = train_data.groupby('month')['Balance'].mean().to_dict()

    train_data['weekday_average'] = list(map(weekday_mean.get, train_data['weekday']))
    train_data['month_average'] = list(map(month_mean.get, train_data['month']))

    # Удаляем ненужные столбцы
    train_data.drop(['weekday', 'month'], axis=1, inplace=True)

    # Удаляем строки с NaN (например, из-за лагов)
    train_data = train_data.dropna().reset_index(drop=True)

    # Разделяем данные на признаки и целевую переменную
    X_train = train_data.drop(["Balance"], axis=1)
    y_train = train_data["Balance"]

    return X_train, y_train, weekday_mean, month_mean

def forecast_method(data, lags, model, weekday_mean, month_mean, column_sort, days_ahead, drift_dates, calendar_extractor):
    data = pd.DataFrame(data.copy())
    last_date = data.index[-1]
    start_date_train = data.index[1]

    # История лагов
    max_lag = max(lags)
    history = list(data['Balance'])[-max_lag:]

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq='D')

    test_predictions = []
    test_features_list = []

    for current_date in future_dates:
        row = {}

        # === Генерация one-hot фичей режима ===
        regime_id = get_regime_for_date(current_date, drift_dates)
        n_regimes = len(drift_dates) + 1
        for i in range(n_regimes):
            row[f'reg_{i}'] = int(i == regime_id)

        # Лаги на основе history
        for ll in lags:
            row[f'lag_{ll}'] = history[-ll]

        for wind in [30,42,90]:
          row[f'sma_{wind}'] = np.mean(history[-wind:]) if len(history) >= wind else 0
          row[f'std_{wind}'] = np.std(history[-wind:]) if len(history) >= wind else 0
        #   row[f'min_{wind}'] = np.min(history[-wind:]) if len(history) >= wind else 0
        #   row[f'max_{wind}'] = np.max(history[-wind:]) if len(history) >= wind else 0


        period = 66.7
        t = (current_date - start_date_train).days
        row['sin_mid_term'] = np.sin(2 * np.pi * t / period)
        row['cos_mid_term'] = np.cos(2 * np.pi * t / period)


        # Категориальные
        weekday = current_date.weekday()
        month = current_date.month
        row['weekday_average'] = weekday_mean.get(weekday, 0)
        row['month_average'] = month_mean.get(month, 0)

        df_test = calendar_extractor.get_calendar_features_for_date(current_date)
        df_test_fixed = calendar_extractor.restore_missing_dummies(df_test, ['is_weekends', 'is_preholiday', 'day_of_month',
       'day_of_year', 'week_of_year', 'is_month_start', 'is_month_end', 'wd_0',
       'wd_1', 'wd_2', 'wd_3', 'wd_4', 'wd_5', 'wd_6', 'm_1', 'm_2', 'm_3',
       'm_4', 'm_5', 'm_6', 'm_7', 'm_8', 'm_9', 'm_10', 'm_11', 'm_12', 'q_1',
       'q_2', 'q_3', 'q_4', 's_autumn', 's_spring', 's_summer', 's_winter',
       'is_25','is_26','is_9','is_10','is_5','is_27', 'is_tax_week'])

        # Создаём датафрейм одной строки
        row_df = pd.DataFrame([row])
        row_df = pd.concat([row_df.reset_index(drop=True), df_test_fixed.reset_index(drop=True)], axis=1)

        # Предсказание
        if (row_df['is_weekends']==1).all():
          prediction = 0
        else:
          row_df = row_df.reindex(columns=column_sort, fill_value=0)
          prediction = model.predict(row_df)[0]
        test_predictions.append(prediction)
        test_features_list.append(row)

        # Обновляем историю значений (добавляем предсказание)
        history.append(prediction)

    # Финальный DataFrame с фичами и предсказаниями
    test_features_df = pd.DataFrame(test_features_list)
    test_features_df['predicted_Balance'] = test_predictions
    test_features_df.index = future_dates

    return test_features_df
