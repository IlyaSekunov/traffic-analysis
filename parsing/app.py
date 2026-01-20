#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import sys


class DataHandler(ABC):
    """Базовый класс обработчика в цепочке ответственности"""
    
    def __init__(self):
        self._next_handler = None
    
    def set_next(self, handler: 'DataHandler') -> 'DataHandler':
        self._next_handler = handler
        return handler
    
    @abstractmethod
    def handle(self, data: pd.DataFrame, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        if self._next_handler:
            return self._next_handler.handle(data, context)
        return data


class DataLoaderHandler(DataHandler):
    """Загрузка CSV данных"""
    
    def handle(self, data: pd.DataFrame, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        if data is None:
            try:
                file_path = context['file_path']
                print(f"Загрузка данных из {file_path}...")
                data = pd.read_csv(file_path, low_memory=False)
                context['original_shape'] = data.shape
                print(f"+ Данные загружены. Размер: {data.shape}")
            except Exception as e:
                print(f"- Ошибка загрузки: {e}")
                sys.exit(1)
        return super().handle(data, context)


class CleanDataHandler(DataHandler):
    """Очистка данных"""
    
    def handle(self, data: pd.DataFrame, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        initial_rows = len(data)
        
        # Удаление полностью пустых строк
        data = data.dropna(how='all')
        # Удаление дубликатов
        data = data.drop_duplicates()
        
        removed = initial_rows - len(data)
        if removed > 0:
            print(f"✓ Очистка данных. Удалено строк: {removed}")
        
        context['cleaned_shape'] = data.shape
        return super().handle(data, context)


class FeatureSelectionHandler(DataHandler):
    """Выбор признаков"""
    
    def handle(self, data: pd.DataFrame, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        # Удаляем неинформативные столбцы
        cols_to_drop = []
        
        # Удаляем столбцы с уникальными значениями (ID и т.д.)
        for col in data.columns:
            if data[col].nunique() == len(data):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            print(f"+Удалены столбцы с уникальными значениями: {cols_to_drop}")
        
        context['selected_features'] = list(data.columns)
        return super().handle(data, context)


class SplitDataHandler(DataHandler):
    """Разделение на X и y"""
    
    def handle(self, data: pd.DataFrame, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        if len(data.columns) < 2:
            raise ValueError("Недостаточно столбцов для разделения")
        
        # Последний столбец как целевая переменная
        target_col = data.columns[-1]
        
        y = data[target_col].values
        X = data.drop(columns=[target_col]).values
        
        context['X_data'] = X
        context['y_data'] = y
        context['target_column'] = target_col
        
        print(f"+ Данные разделены. X: {X.shape}, y: {y.shape}")
        print(f"  Целевая переменная: '{target_col}'")
        
        return super().handle(data, context)


class SaveNumpyHandler(DataHandler):
    """Сохранение в .npy"""
    
    def handle(self, data: pd.DataFrame, context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        output_dir = context.get('output_dir', '.')
        
        try:
            np.save(f"{output_dir}/x_data.npy", context['X_data'])
            np.save(f"{output_dir}/y_data.npy", context['y_data'])
            print(f"+ Файлы сохранены:")
            print(f"  - {output_dir}/x_data.npy")
            print(f"  - {output_dir}/y_data.npy")
        except Exception as e:
            print(f"- Ошибка сохранения: {e}")
            sys.exit(1)
        
        return super().handle(data, context)


class DataProcessingPipeline:
    """Пайплайн обработки данных"""
    
    def __init__(self):
        # Создаем цепочку обработчиков
        self.loader = DataLoaderHandler()
        self.cleaner = CleanDataHandler()
        self.feature_selector = FeatureSelectionHandler()
        self.splitter = SplitDataHandler()
        self.saver = SaveNumpyHandler()
        
        # Настраиваем цепочку
        self.loader.set_next(self.cleaner) \
                  .set_next(self.feature_selector) \
                  .set_next(self.splitter) \
                  .set_next(self.saver)
    
    def process(self, file_path: str, output_dir: str = ".") -> None:
        """Запуск пайплайна обработки"""
        print(f"\n{'='*50}")
        print("Запуск пайплайна обработки данных")
        print(f"{'='*50}")
        
        context = {
            'file_path': file_path,
            'output_dir': output_dir
        }
        
        try:
            self.loader.handle(None, context)
            print(f"\n{'='*50}")
            print("Обработка завершена успешно!")
            print(f"{'='*50}")
        except Exception as e:
            print(f"\n✗ Ошибка в пайплайне: {e}")
            sys.exit(1)


def main():
    """Точка входа в программу"""
    parser = argparse.ArgumentParser(
        description='Обработка данных HH с применением паттерна "Цепочка ответственности"'
    )
    parser.add_argument(
        'file_path',
        type=str,
        help='Путь к CSV файлу с данными'
    )
    
    args = parser.parse_args()
    
    # Создаем и запускаем пайплайн
    pipeline = DataProcessingPipeline()
    pipeline.process(args.file_path)


if __name__ == "__main__":
    main()