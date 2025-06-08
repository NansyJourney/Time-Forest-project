# Time-Forest-project
Automated pipeline for predicting bank balance

## Структура проекта
```
Time-Forest-project/
│
├── data/                # данные проекта  
│   ├── Data.xlsx        # исходные данные  
│   ├── ключ ставка.xls  # ключевая ставка
│
├── src/
│   ├── AnomalyDetector.py            # Модуль для поиска аномальных значений
│   ├── CalendarExtractor.py          # Модуль для парсинга фичей из календаря  
│   ├── ChangeDetection.py            # Модуль для определения разладок 
│   ├── FeatureExtractor.py           # Модуль для парсинга фичей из временного ряда   
│   ├── TrainForecasterModel.py       # Модуль для обучения и предсказания моделей
│
├── Проект по предсказанию баланса.ipynb #Сам проект

