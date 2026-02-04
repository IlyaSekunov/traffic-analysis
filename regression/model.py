"""
Модель для предсказания зарплат.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

    def __init__(self, x_train_path=None, y_train_path=None):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.x_train_path = x_train_path or TRAIN_X_FILE
        self.y_train_path = y_train_path or TRAIN_Y_FILE

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
        print(f"✅ Модель загружена из {MODEL_PATH}")

    def _train(self):
        """Обучает модель на указанных данных."""
        print(f"📥 Загрузка данных для обучения:")
        print(f"   X: {self.x_train_path}")
        print(f"   Y: {self.y_train_path}")

        # Загрузка данных для обучения
        X = np.load(self.x_train_path)
        y = np.load(self.y_train_path)

        # Обработка NaN
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        print(f"📊 Данные для обучения: {X.shape[0]} образцов, {X.shape[1]} признаков")

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )

        # Масштабирование
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Обучение модели
        print("🧠 Обучение RandomForestRegressor...")
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
        print(f"📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
        print(f"   R²: {r2}")
        print(f"   MSE: {mse}")

        # Сохранение модели
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

        print(f"💾 Модель сохранена в {RESOURCES_DIR}")

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
        print(f"📥 Загружено {len(X)} образцов для предсказания")

        # Обработка NaN (замена на средние)
        if np.isnan(X).any():
            print("⚠️  Обнаружены NaN значения, заменяю на средние...")
            col_means = np.nanmean(X, axis=0)
            nan_indices = np.where(np.isnan(X))
            X[nan_indices] = np.take(col_means, nan_indices[1])

        # Масштабирование
        X_scaled = self.scaler.transform(X)

        # Предсказание
        predictions = self.model.predict(X_scaled)

        # Округление до 2 знаков (копейки)
        predictions = np.round(predictions, 2)

        print(f"✅ Сделано {len(predictions)} предсказаний")

        return predictions.tolist()


# Дополнительная функция для обучения с указанием путей
def train_model(x_path, y_path):
    """
    Обучает модель на указанных путях к данным.

    Args:
        x_path: Путь к файлу X.npy
        y_path: Путь к файлу y.npy

    Returns:
        Обученная модель
    """
    model = SalaryModel(x_train_path=x_path, y_train_path=y_path)
    model.load_or_train()  # Это обучит новую модель
    return model