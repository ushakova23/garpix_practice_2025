import pandas as pd
from src.wb_ranking_model import WBRankingModel
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Обучение полной LightGBM модели для WB')
    parser.add_argument('data_path', type=str,
                       help='Путь к файлу с данными (CSV, TSV, Excel)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Размер тестовой выборки (по умолчанию 0.2)')
    
    args = parser.parse_args()
    
    print("Запускаем обучение LightGBM модели")
    print(f"Загружаем данные из: {args.data_path}")
    
    try:
        if args.data_path.endswith('.csv'):
            df = pd.read_csv(args.data_path)
        elif args.data_path.endswith('.tsv'):
            df = pd.read_csv(args.data_path, delimiter='\t')
        elif args.data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(args.data_path)
        else:
            raise ValueError("Поддерживаются только форматы: CSV, TSV, Excel")
    except Exception as e:
        print(f"Не удалось загрузить данные: {e}")
        sys.exit(1)
    
    print(f"Загружено {len(df)} записей")
    print(f"Колонки: {list(df.columns)}")
    
    # Проверяем ключевые колонки
    required_columns = ['position', 'proceeds']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Отсутствуют обязательные колонки: {missing_columns}")
        print("Доступные колонки:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        sys.exit(1)
    
    # Статистика по ключевым полям
    print(f"\nСтатистика по позициям:")
    print(f"  Валидных позиций: {(df['position'] > 0).sum()}")
    print(f"  Нулевых/отрицательных: {(df['position'] <= 0).sum()}")
    print(f"  Пропущенных позиций: {df['position'].isna().sum()}")
    
    print(f"\nСтатистика по выручке:")
    print(f"  Валидных записей: {(df['proceeds'] >= 0).sum()}")
    print(f"  Отрицательных: {(df['proceeds'] < 0).sum()}")
    print(f"  Пропущенной выручки: {df['proceeds'].isna().sum()}")
    
    print(f"\nПримеры данных:")
    print(df[['position', 'proceeds']].head())
    
    model = WBRankingModel()
    
    print("\nНачинаем обучение моделей...")
    try:
        model.train_models(df, test_size=args.test_size)
    except Exception as e:
        print(f"Ошибка обучения: {e}")
        sys.exit(1)
    
    print("\nАнализируем важность признаков:")
    try:
        model.analyze_feature_importance()
    except Exception as e:
        print(f"Ошибка анализа важности: {e}")
    
    print("\nСохраняем модели...")
    try:
        model.save_models()
    except Exception as e:
        print(f"Ошибка сохранения: {e}")
        sys.exit(1)
    
    print("\nОбучение завершено! Модели сохранены.")
    print("Файлы моделей:")
    print("- position_model.pkl (модель позиции)")
    print("- proceeds_model.pkl (модель выручки)")
    print("- feature_list.pkl (список признаков)")
    import json

    print("\nСохраняем важность признаков...")
    try:
        importance_data_numpy = model.get_feature_importance_dict()
    
        importance_data_python = {key: int(value) for key, value in importance_data_numpy.items()}
    
        with open("models/feature_importance.json", "w", encoding="utf-8") as f:
            json.dump(importance_data_python, f, ensure_ascii=False, indent=4)
        
        print("Важность признаков сохранена в models/feature_importance.json")
    except Exception as e:
        print(f"Ошибка сохранения важности признаков: {e}")

    print("\nОбучение завершено!")


if __name__ == "__main__":
    main()
