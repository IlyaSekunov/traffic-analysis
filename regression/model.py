"""
Модель для предсказания зарплат.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

from config import (
    MODEL_PATH,
    SCALER_PATH,
    RESOURCES_DIR,
    TRAIN_X_FILE,
    TRAIN_Y_FILE,
    TEST_SIZE,
    RANDOM_STATE,
)


class SalaryModel:
    """Модель для предсказания зарплат."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False

    def load_or_train(self):
        """Загружает модель или обучает новую если нет сохраненной."""
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            self._load()
        else:
            self._train()

    def _load(self):
        """Загружает сохраненную модель."""
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.is_trained = True

    def _train(self):
        """Обучает модель на данных из папки parsing."""
        # Загрузка данных для обучения
        X = np.load(TRAIN_X_FILE)
        y = np.load(TRAIN_Y_FILE)

        # Обработка NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )

        # Масштабирование
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучение модели
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Оценка модели и вывод метрик
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # Вывод метрик качества
        print(f"R²: {r2}")
        print(f"MSE: {mse}")

        # Сохранение модели
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

    def predict(self, X_path: str) -> list:
        """
        Предсказывает зарплаты для данных из файла.

        Args:
            X_path: Путь к файлу .npy с данными для предсказания

        Returns:
            Список зарплат в рублях
        """
        if not self.is_trained:
            self.load_or_train()

        # Загрузка данных для предсказания
        X = np.load(X_path)

        # Обработка NaN (замена на средние)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            nan_indices = np.where(np.isnan(X))
            X[nan_indices] = np.take(col_means, nan_indices[1])

        # Масштабирование
        X_scaled = self.scaler.transform(X)

        # Предсказание
        predictions = self.model.predict(X_scaled)

        # Округление до 2 знаков (копейки)
        predictions = np.round(predictions, 2)

        return predictions.tolist()