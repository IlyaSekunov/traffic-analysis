"""
Конфигурация приложения.
"""

import sys
from pathlib import Path
from typing import Final

# Пути проекта
BASE_DIR: Final = Path(__file__).resolve().parent.parent
REGRESSION_DIR: Final = Path(__file__).resolve().parent
PARSING_DIR: Final = BASE_DIR / "parsing"
RESOURCES_DIR: Final = REGRESSION_DIR / "resources"

# Создание папки resources при необходимости
RESOURCES_DIR.mkdir(exist_ok=True)

# Имена файлов для обучения (по умолчанию)
TRAIN_X_FILE: Final = PARSING_DIR / "x_data.npy"
TRAIN_Y_FILE: Final = PARSING_DIR / "y_data.npy"

# Имена файлов моделей
MODEL_FILENAME: Final = "trained_model.joblib"
SCALER_FILENAME: Final = "scaler.joblib"
MODEL_PATH: Final = RESOURCES_DIR / MODEL_FILENAME
SCALER_PATH: Final = RESOURCES_DIR / SCALER_FILENAME

# Параметры обучения
TEST_SIZE: Final = 0.2
RANDOM_STATE: Final = 42