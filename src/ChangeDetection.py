import ruptures as rpt
import numpy as np

def detect_pelt(data, pen=2, min_size=5, model='rbf'):
    algo = rpt.Pelt(model=model, min_size=min_size).fit(data)
    return algo.predict(pen=pen)[:-1]  # без последней точки конца

def detect_cusum_dual(data, threshold=5):
    s_pos, s_neg = 0, 0
    change_points = []
    mean = np.mean(data)
    for i in range(1, len(data)):
        delta = data[i] - mean
        s_pos = max(0, s_pos + delta)
        s_neg = min(0, s_neg + delta)
        if s_pos > threshold or s_neg < -threshold:
            change_points.append(i)
            s_pos = s_neg = 0
    return change_points

def confirm_consensus(cps_pelt, cps_cusum, tolerance=5):
    confirmed = []
    for p_cp in cps_pelt:
        for c_cp in cps_cusum:
            if abs(p_cp - c_cp) <= tolerance:
                confirmed.append(p_cp)
                break
    return confirmed


def check_new_drift(full_series, old_detected=None, tolerance=7,
                    pelt_params=None, cusum_params=None, window_size=300):
    """
    Проверяет, появилась ли новая согласованная разладка на новых данных
    :param full_series: полный pd.Series / np.array всех данных
    :param old_detected: список ранее найденных разладок
    :param tolerance: допустимое расстояние между согласованными точками
    :param pelt_params: dict с настройками PELT
    :param cusum_params: dict с настройками CUSUM
    :param window_size: размер скользящего окна
    :return: флаг, новая разладка (bool), список новых точек
    """
    if pelt_params is None:
        pelt_params = {"pen": 2, "min_size": 5, "model": "rbf"}
    if cusum_params is None:
        cusum_params = {"threshold": 2}

    # Используем скользящее окно с сбросом индексов
    recent_data = full_series[-window_size:]

    # Детекция
    cps_pelt = detect_pelt(recent_data, **pelt_params)
    cps_cusum = detect_cusum_dual(recent_data, **cusum_params)

    # Согласование
    confirmed = confirm_consensus(cps_pelt, cps_cusum, tolerance=tolerance)

    # Преобразуем локальные индексы в глобальные
    confirmed_global = [len(full_series) - window_size + cp for cp in confirmed]

    # Исключаем уже известные и фильтруем по времени
    if old_detected is None:
        new_confirmed = confirmed_global
    else:
      # Находим самую позднюю ранее обнаруженную разладку
      latest_old_detected = max(old_detected)
      # Фильтруем только те разладки, которые больше latest_old_detected
      new_confirmed = [cp for cp in confirmed_global if cp > latest_old_detected]

    drift_detected = len(new_confirmed) > 0
    return drift_detected, new_confirmed


def filter_drifts(drift_points, min_distance):
    filtered = []
    for point in drift_points:
        if not filtered or point - filtered[-1] >= min_distance:
            filtered.append(point)
    return filtered

def change_detection(data):
    cps_pelt = detect_pelt(data, pen=2, min_size=5, model='rbf')
    cps_cusum_dual = detect_cusum_dual(data, threshold=2)
    return confirm_consensus(cps_pelt, cps_cusum_dual, tolerance=15)
