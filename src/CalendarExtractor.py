import pandas as pd

class CalendarFeatureExtractor:
    def __init__(self, calendar_path="https://raw.githubusercontent.com/szonov/data-gov-ru-calendar/master/calendar.csv", year_range=(2017, 2022)):
        """
        Инициализирует CalendarFeatureExtractor, загружая и фильтруя календарь.

        Args:
            calendar_path (str): Путь к CSV файлу календаря.
            year_range (tuple): Диапазон лет для фильтрации (начальный год, конечный год).
        """
        self.calendar_df = self._load_and_filter_calendar(calendar_path, year_range)
        self.calendar_parsed_df = self._parse_calendar(self.calendar_df)

    def _load_and_filter_calendar(self, path, year_range):
        """
        Загружает календарь из CSV файла, фильтрует по диапазону лет и выбирает нужные месяцы (приватный метод).
        """
        months = ['Год/Месяц', 'Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                  'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
        calendar = pd.read_csv(path)
        calendar = calendar[(calendar['Год/Месяц'] >= year_range[0]) & (calendar['Год/Месяц'] <= year_range[1])]
        calendar = calendar[months]
        return calendar

    def _parse_calendar(self, calendar):
        """
        Преобразует DataFrame календаря в DataFrame с датами и признаками выходных и предпраздничных дней.
        """
        date_list = []

        for _, row in calendar.iterrows():
            year = int(row['Год/Месяц'])
            for month_num, month_name in enumerate(calendar.columns[1:], start=1):
                days_raw = str(row[month_name])
                if days_raw and days_raw != 'nan':
                    days = [d.strip() for d in days_raw.split(',') if d.strip()]
                    for day in days:
                        if '+' in day:
                            continue  # пропускаем бессмысленные

                        is_preholiday = '*' in day
                        day_clean = int(day.replace('*', ''))

                        try:
                            date = pd.Timestamp(year=year, month=month_num, day=day_clean)
                            date_list.append({
                                'date': date,
                                'is_weekends': int(not is_preholiday),  # если нет *, то это выходной
                                'is_preholiday': int(is_preholiday)
                            })
                        except ValueError:
                            continue  # пропуск некорректных дат

        df = pd.DataFrame(date_list).drop_duplicates(subset=['date']).set_index('date')

        # Гарантируем, что один день не может быть и выходным, и предпраздничным одновременно
        df.loc[df['is_preholiday'] == 1, 'is_weekends'] = 0

        return df

    def enrich_with_calendar_features(self, df):
        """
        Обогащает DataFrame с датами календарными признаками, используя предварительно обработанный календарь.

        Args:
            df (pandas.DataFrame): DataFrame с датами в качестве индекса.

        Returns:
            pandas.DataFrame: DataFrame с добавленными календарными признаками.
        """
        calendar_df = self.calendar_parsed_df
        df = df.join(calendar_df, how='left')
        df[['is_weekends', 'is_preholiday']] = df[['is_weekends', 'is_preholiday']].fillna(0)

        df['weekday'] = df.index.weekday
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_month'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)

        def get_season(month):
            if month in [12, 1, 2]: return 'winter'
            if month in [3, 4, 5]: return 'spring'
            if month in [6, 7, 8]: return 'summer'
            if month in [9, 10, 11]: return 'autumn'

        df['season'] = df.index.month.map(get_season)
        df = pd.get_dummies(df, columns=['weekday', 'month', 'quarter', 'season'],
                            prefix=['wd', 'm', 'q', 's'], drop_first=False)

        df['is_25'] = df.index.day.isin([25]).astype(int)
        df['is_26'] = df.index.day.isin([26]).astype(int)
        df['is_9'] = df.index.day.isin([9]).astype(int)
        df['is_10'] = df.index.day.isin([10]).astype(int)
        df['is_5'] = df.index.day.isin([5]).astype(int)
        df['is_27'] = df.index.day.isin([27]).astype(int)
        df['is_tax_week'] = df.index.day.map(lambda x: 21 <= x <= 30).astype(int)

        # Переводим bool в int
        bool_cols = df.select_dtypes(include=['bool']).columns
        if not bool_cols.empty:
            df[bool_cols] = df[bool_cols].astype(int)

        return df.fillna(0)

    def get_calendar_features_for_date(self, date_input):
        """
        Возвращает календарные признаки для переданной даты или списка дат, используя предварительно обработанный календарь.

        Args:
            date_input (str, datetime, list): Строка, datetime объект или список дат.

        Returns:
            pandas.DataFrame: DataFrame с календарными признаками для запрошенных дат.
        """
        # Приводим к списку дат
        if isinstance(date_input, (str, pd.Timestamp)):
            dates = [pd.to_datetime(date_input)]
        else:
            dates = pd.to_datetime(date_input)

        df = pd.DataFrame(index=pd.DatetimeIndex(dates))
        return self.enrich_with_calendar_features(df)

    def restore_missing_dummies(self, df_new, required_columns):
        """
        Добавляет недостающие one-hot колонки и выравнивает порядок.

        Args:
            df_new (pandas.DataFrame): DataFrame с текущими фичами.
            required_columns (list or set): Список или набор всех нужных столбцов.

        Returns:
            pandas.DataFrame: DataFrame с добавленными нулями и правильным порядком столбцов.
        """
        df = df_new.copy()
        missing_cols = set(required_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        return df[list(required_columns)]