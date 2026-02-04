## 📁 Структура проекта

```
traffic_analysis/
├── parsing/                    # Исходные данные для обучения
│   ├── x_data.npy             # Признаки для обучения (по умолчанию)
│   └── y_data.npy             # Целевые значения для обучения (по умолчанию)
├── regression/
│   ├── app.py                 # Основной скрипт с CLI интерфейсом
│   ├── model.py               # Модель машинного обучения
│   ├── config.py              # Конфигурация приложения
│   ├── utils.py               # Вспомогательные функции
│   ├── requirements.txt       # Зависимости проекта
│   ├── README.md              # Эта документация
│   └── resources/             # Папка для сохранения весов модели
│       ├── trained_model.joblib
│       └── scaler.joblib
└── .gitignore
```

## 🚀 Установка

1. Клонируйте репозиторий или скопируйте файлы проекта
2. Перейдите в папку проекта:
   ```bash
   cd traffic_analysis/regression
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
   
   Если файла requirements.txt нет, установите зависимости вручную:
   ```bash
   pip install numpy scikit-learn joblib
   ```

## 📊 Подготовка данных

Для обучения модели подготовьте два файла в формате `.npy`:

1. **Признаки (X)**: матрица формата `(n_samples, n_features)` в файле `x_data.npy`
2. **Целевые значения (y)**: вектор формата `(n_samples,)` в файле `y_data.npy` (зарплаты в рублях)

По умолчанию модель ищет эти файлы в папке `../parsing/` относительно папки `regression`.

## 🎯 Использование

### 1. Предсказание зарплат (основной сценарий)

```bash
# Использование файлов по умолчанию из папки parsing
python app.py predict ../parsing/x_data.npy

# Использование своих данных
python app.py predict path/to/your_data.npy
```

**Что происходит:**
- Если модель уже обучена и сохранена в `resources/`, она загружается
- Если модель не обучена, она обучается на данных по умолчанию из `parsing/`
- Делаются предсказания для входных данных
- Вывод: список зарплат в рублях (float), по одному на строку

**Пример вывода:**
```
75000.50
72000.00
85000.25
...
```

### 2. Обучение модели на своих данных

```bash
# Обучение на данных из папки parsing
python app.py train ../parsing/x_data.npy ../parsing/y_data.npy

# Обучение на своих данных
python app.py train path/to/X_train.npy path/to/y_train.npy
```

**Что происходит:**
- Удаляется предыдущая модель (если есть)
- Обучается новая модель на указанных данных
- Выводятся метрики качества (R² и MSE)
- Модель сохраняется в папку `resources/`

**Пример вывода:**
```
🧠 Обучение модели на данных:
   X: ../parsing/x_data.npy
   Y: ../parsing/y_data.npy
📥 Загрузка данных для обучения:
   X: ../parsing/x_data.npy
   Y: ../parsing/y_data.npy
📊 Данные для обучения: 1000 образцов, 20 признаков
🧠 Обучение RandomForestRegressor...
📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:
   R²: 0.65
   MSE: 800000.0
💾 Модель сохранена в traffic_analysis/regression/resources
✅ Модель успешно обучена и сохранена
```

### 3. Получение справки

```bash
# Общая справка
python app.py --help

# Справка по команде predict
python app.py predict --help

# Справка по команде train
python app.py train --help
```

## 💾 Сохранение модели

После обучения модель сохраняется в папку `resources/`:
- `trained_model.joblib` - обученная модель
- `scaler.joblib` - параметры масштабирования

При следующем запуске `predict` модель загружается из этих файлов, а не обучается заново.

## 🔄 Переобучение модели

Чтобы переобучить модель на новых данных:

```bash
# Удалить сохраненную модель
rm -rf traffic_analysis/regression/resources

# Или использовать команду train
python app.py train ../parsing/x_data.npy ../parsing/y_data.npy
```

## 🐛 Возможные проблемы и решения

### 1. Файлы данных не найдены
```
❌ Файл не найден: ../parsing/x_data.npy
```
**Решение**: Убедитесь, что файлы существуют по указанному пути.

### 2. Ошибка импорта
```
ModuleNotFoundError: No module named 'sklearn'
```
**Решение**: Установите scikit-learn:
```bash
pip install scikit-learn
```

### 3. Неправильный формат файлов
```
ValueError: Unknown file format
```
**Решение**: Убедитесь, что файлы имеют расширение `.npy` и созданы с помощью `numpy.save()`.

### 4. Разные размеры данных
```
ValueError: X and y have different number of samples
```
**Решение**: Убедитесь, что количество строк в X и y совпадает.

## 📝 Пример полного рабочего процесса

```bash
# 1. Перейти в папку проекта
cd traffic_analysis/regression

# 2. Установить зависимости
pip install numpy scikit-learn joblib

# 3. Поместить данные в папку parsing
#    (убедитесь, что есть файлы x_data.npy и y_data.npy)

# 4. Обучить модель
python app.py train ../parsing/x_data.npy ../parsing/y_data.npy

# 5. Сделать предсказания
python app.py predict ../parsing/x_data.npy

# 6. Сохранить предсказания в файл
python app.py predict ../parsing/x_data.npy > predictions.txt
```