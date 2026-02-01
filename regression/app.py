"""
Основной скрипт приложения.

Использование:
    python app.py path/to/x_data.npy
"""

import sys
from pathlib import Path
from model import SalaryModel


def main():
    # Проверка аргументов
    if len(sys.argv) != 2:
        print("Использование: python app.py path/to/x_data.npy")
        sys.exit(1)

    x_path = Path(sys.argv[1])

    # Проверка существования файла
    if not x_path.exists():
        print(f"❌ Файл не найден: {x_path}")
        sys.exit(1)

    # Проверка расширения файла
    if x_path.suffix != '.npy':
        print("❌ Файл должен иметь расширение .npy")
        sys.exit(1)

    # Создание и использование модели
    model = SalaryModel()

    try:
        predictions = model.predict(str(x_path))

        # Вывод результатов (по одному числу на строку)
        #for salary in predictions:
         #   print(salary)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()