"""
Основной скрипт приложения.

Использование:
    python app.py predict path/to/x_data.npy
    python app.py train path/to/x_data.npy path/to/y_data.npy
"""

import sys
from pathlib import Path
from model import SalaryModel
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Предсказание зарплат на основе данных hh.ru",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python app.py predict ../parsing/X.npy
  python app.py train ../parsing/X.npy ../parsing/y.npy
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # Парсер для команды predict
    predict_parser = subparsers.add_parser('predict', help='Предсказание зарплат')
    predict_parser.add_argument(
        'x_path',
        type=Path,
        help='Путь к файлу .npy с признаками для предсказания'
    )

    # Парсер для команды train
    train_parser = subparsers.add_parser('train', help='Обучение модели')
    train_parser.add_argument(
        'x_path',
        type=Path,
        help='Путь к файлу .npy с признаками для обучения'
    )
    train_parser.add_argument(
        'y_path',
        type=Path,
        help='Путь к файлу .npy с целевыми значениями для обучения'
    )

    args = parser.parse_args()

    if args.command == 'predict':
        predict(args.x_path)
    elif args.command == 'train':
        train(args.x_path, args.y_path)
    else:
        parser.print_help()
        sys.exit(1)


def predict(x_path: Path):
    """Выполняет предсказание зарплат."""
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
        for salary in predictions:
            print(salary)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


def train(x_path: Path, y_path: Path):
    """Обучает модель на указанных данных."""
    # Проверка существования файлов
    if not x_path.exists():
        print(f"❌ Файл не найден: {x_path}")
        sys.exit(1)

    if not y_path.exists():
        print(f"❌ Файл не найден: {y_path}")
        sys.exit(1)

    # Проверка расширения файлов
    if x_path.suffix != '.npy':
        print("❌ Файл X должен иметь расширение .npy")
        sys.exit(1)

    if y_path.suffix != '.npy':
        print("❌ Файл Y должен иметь расширение .npy")
        sys.exit(1)

    # Импортируем конфигурацию для обновления путей
    import config
    from pathlib import Path

    # Сохраняем оригинальные пути
    original_x_path = config.TRAIN_X_FILE
    original_y_path = config.TRAIN_Y_FILE

    try:
        # Временно меняем пути в конфигурации
        config.TRAIN_X_FILE = Path(x_path)
        config.TRAIN_Y_FILE = Path(y_path)

        # Создание и обучение модели
        model = SalaryModel()

        # Принудительно запускаем обучение
        print(f"🧠 Обучение модели на данных:")
        print(f"   X: {x_path}")
        print(f"   Y: {y_path}")

        # Удаляем существующую модель, чтобы принудительно переобучить
        import shutil
        from config import RESOURCES_DIR

        if RESOURCES_DIR.exists():
            print("🗑️  Удаляю предыдущую модель...")
            shutil.rmtree(RESOURCES_DIR)

        # Создаем новую папку resources
        RESOURCES_DIR.mkdir(exist_ok=True)

        # Обучаем модель
        model = SalaryModel()

        # Для обучения вызываем load_or_train, который обучит новую модель
        # так как мы удалили предыдущую
        model.load_or_train()

        print("✅ Модель успешно обучена и сохранена")

    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        sys.exit(1)
    finally:
        # Восстанавливаем оригинальные пути
        config.TRAIN_X_FILE = original_x_path
        config.TRAIN_Y_FILE = original_y_path


if __name__ == "__main__":
    main()