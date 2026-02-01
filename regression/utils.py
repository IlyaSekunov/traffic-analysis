"""
Вспомогательные функции.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    SCALER_PATH,
    RESOURCES_DIR,
    MODEL_PARAMS,
)


def setup_logger(name: str = __name__, verbose: bool = False) -> logging.Logger:
    """Настройка логгера."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def load_data(
    x_path: Path,
    y_path: Optional[Path] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загрузка данных из .npy файлов.

    Args:
        x_path: Путь к файлу с признаками
        y_path: Путь к файлу с целевыми значениями (опционально)
        verbose: Подробный вывод

    Returns:
        Кортеж (X, y)

    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если данные некорректны
    """
    logger = setup_logger(__name__, verbose)

    try:
        X = np.load(x_path)
        logger.debug(f"Загружены признаки из {x_path}. Shape: {X.shape}")
    except FileNotFoundError:
        logger.error(f"Файл {x_path} не найден")
        raise

    if y_path is None:
        # Предполагаем, что y хранится в том же каталоге с именем y.npy
        y_path = x_path.parent / "y.npy"

    try:
        y = np.load(y_path)
        logger.debug(f"Загружены целевые значения из {y_path}. Shape: {y.shape}")
    except FileNotFoundError:
        logger.error(f"Файл {y_path} не найден")
        raise

    # Проверка совместимости размеров
    if len(X) != len(y):
        raise ValueError(
            f"Несовместимые размеры: X.shape[0]={len(X)}, y.shape[0]={len(y)}"
        )

    # Проверка на NaN значения
    nan_info_printed = False

    if np.isnan(X).any():
        if not nan_info_printed:
            logger.info("Обнаружены NaN значения. Обрабатываю...")
            nan_info_printed = True
        # Удаляем строки с NaN
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        y_clean = y[mask]
        logger.debug(f"Удалено {len(X) - len(X_clean)} строк с NaN в X")
        X, y = X_clean, y_clean

    if np.isnan(y).any():
        if not nan_info_printed:
            logger.info("Обнаружены NaN значения. Обрабатываю...")
            nan_info_printed = True
        # Удаляем строки с NaN
        mask = ~np.isnan(y)
        X_clean = X[mask]
        y_clean = y[mask]
        logger.debug(f"Удалено {len(X) - len(X_clean)} строк с NaN в y")
        X, y = X_clean, y_clean

    if nan_info_printed:
        logger.info(f"После обработки NaN: X.shape={X.shape}, y.shape={y.shape}")

    # Проверка на Inf значения
    if np.isinf(X).any():
        logger.warning("Обнаружены бесконечные значения в X")
        X = np.nan_to_num(X, posinf=np.nanmax(X), neginf=np.nanmin(X))

    if np.isinf(y).any():
        logger.warning("Обнаружены бесконечные значения в y")
        y = np.nan_to_num(y, posinf=np.nanmax(y), neginf=np.nanmin(y))

    logger.info(f"Данные успешно загружены. X: {X.shape}, y: {y.shape}")

    return X, y


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    save_scaler: bool = False,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Подготовка данных для обучения.

    Args:
        X: Матрица признаков
        y: Вектор целевых значений
        save_scaler: Сохранить скейлер на диск
        verbose: Подробный вывод

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    logger = setup_logger(__name__, verbose)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    logger.info(f"Разделение данных: train={len(X_train)}, test={len(X_test)}")

    # Масштабирование признаков
    logger.debug("Масштабирование признаков...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Признаки успешно масштабированы")

    # Сохранение скейлера
    if save_scaler:
        save_scaler_to_disk(scaler, verbose=verbose)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_model_to_disk(
    model: Any,
    scaler: Optional[StandardScaler] = None,
    verbose: bool = False
) -> None:
    """
    Сохранение обученной модели и скейлера в папку resources.

    Args:
        model: Обученная модель
        scaler: Обученный скейлер (опционально)
        verbose: Подробный вывод
    """
    logger = setup_logger(__name__, verbose)

    # Сохранение модели
    try:
        joblib.dump(model, MODEL_PATH)
        logger.info(f"✅ Модель сохранена в {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении модели: {str(e)}")
        raise

    # Сохранение скейлера
    if scaler is not None:
        try:
            joblib.dump(scaler, SCALER_PATH)
            logger.info(f"✅ Скейлер сохранен в {SCALER_PATH}")
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении скейлера: {str(e)}")
            # Не прерываем выполнение, если не удалось сохранить скейлер


def save_scaler_to_disk(scaler: StandardScaler, verbose: bool = False) -> None:
    """
    Сохранение только скейлера.

    Args:
        scaler: Обученный скейлер
        verbose: Подробный вывод
    """
    logger = setup_logger(__name__, verbose)

    try:
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"✅ Скейлер сохранен в {SCALER_PATH}")
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении скейлера: {str(e)}")
        raise


def load_model_from_disk(verbose: bool = False) -> Tuple[Any, Optional[StandardScaler]]:
    """
    Загрузка обученной модели и скейлера из папки resources.

    Returns:
        Кортеж (модель, скейлер)

    Raises:
        FileNotFoundError: Если файлы моделей не найдены
    """
    logger = setup_logger(__name__, verbose)

    if not MODEL_PATH.exists():
        error_msg = f"Файл модели не найден: {MODEL_PATH}"
        logger.error(error_msg)
        logger.info(f"📁 Содержимое папки {RESOURCES_DIR}:")
        for file in RESOURCES_DIR.iterdir():
            logger.info(f"  - {file.name}")
        raise FileNotFoundError(error_msg)

    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Модель загружена из {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке модели: {str(e)}")
        raise

    scaler = None
    if SCALER_PATH.exists():
        try:
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"✅ Скейлер загружен из {SCALER_PATH}")
        except Exception as e:
            logger.warning(f"⚠️  Ошибка при загрузке скейлера: {str(e)}")
    else:
        logger.warning(f"⚠️  Файл скейлера не найден: {SCALER_PATH}")

    return model, scaler


def check_resources_dir() -> bool:
    """
    Проверка существования папки resources и файлов моделей.

    Returns:
        True если папка существует и содержит файлы
    """
    if not RESOURCES_DIR.exists():
        print(f"⚠️  Папка resources не найдена: {RESOURCES_DIR}")
        print(f"📁 Создаю папку...")
        try:
            RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
            print(f"✅ Папка создана: {RESOURCES_DIR}")
        except Exception as e:
            print(f"❌ Ошибка при создании папки: {str(e)}")
            return False

    return True


def get_model_info() -> Dict[str, any]:
    """
    Получение информации о сохраненной модели.

    Returns:
        Словарь с информацией о модели
    """
    info = {
        "resources_dir": str(RESOURCES_DIR),
        "model_exists": MODEL_PATH.exists(),
        "scaler_exists": SCALER_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }

    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            info["model_type"] = type(model).__name__
            if hasattr(model, "n_estimators"):
                info["n_estimators"] = model.n_estimators
            if hasattr(model, "feature_importances_"):
                info["n_features"] = len(model.feature_importances_)
        except:
            info["model_type"] = "Unknown (error loading)"

    return info