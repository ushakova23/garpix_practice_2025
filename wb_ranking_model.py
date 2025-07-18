import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class WBRankingModel:
    def __init__(self):
        self.position_model = None
        self.revenue_model = None
        self.label_encoders = {}
        self.feature_columns = None
        self.data = None
        
    def prepare_data_from_unified_dataset(self, df):
        """
        Подготавливаем данные из единого датасета
        """
        # Создаем копию для обработки
        data = df.copy()
        
        # Создание дополнительных признаков на основе имеющихся данных
        data = self._create_features_from_existing_data(data)
        
        return data
    
    def _create_features_from_existing_data(self, df):
        """
        Создаем полезные признаки из того что у нас есть
        """
        print("Создаем дополнительные признаки из данных...")
        
        # Безопасное деление с обработкой проблемных значений
        def safe_divide(a, b, default=0):
            """Безопасное деление с заменой inf и nan на default"""
            result = a / (b + 0.01)  # добавляем малое число для избежания деления на 0
            result = result.replace([np.inf, -np.inf], default)
            result = result.fillna(default)
            return result
        
        # Базовые признаки эффективности
        df['is_discounted'] = (df.get('discount', 0) > 0).astype(int)
        df['price_per_rating'] = safe_divide(df.get('price', 0), df.get('product_rating', 1))
        
        # Взвешенный рейтинг с проверкой на проблемные значения
        reviews_count = df.get('reviews_count', 0).fillna(0)
        product_rating = df.get('product_rating', 0).fillna(0)
        df['rating_weighted'] = product_rating * np.log1p(reviews_count)
        
        # Признаки доставки и позиций
        delivery_pos = df.get('delivery_efficiency_wh_avg_pos', 0).fillna(0)
        position = df.get('position', 1).fillna(1)
        df['delivery_position_ratio'] = safe_divide(delivery_pos, position)
        df['position_squared'] = position ** 2
        df['position_log'] = np.log1p(position)
        
        # Ценовые признаки
        price = df.get('price', 0).fillna(0)
        discount = df.get('discount', 0).fillna(0)
        df['discount_price_ratio'] = safe_divide(discount, price)
        
        # Создание категорий цен с обработкой проблем
        try:
            df['price_category'] = pd.cut(price, bins=5, labels=['low', 'below_avg', 'avg', 'above_avg', 'high'])
            df['price_category'] = df['price_category'].astype(str).fillna('unknown')
        except Exception:
            df['price_category'] = 'unknown'
        
        # Признаки качества товара
        images_count = df.get('images_count', 0).fillna(0)
        df['has_images'] = (images_count > 0).astype(int)
        df['image_quality_score'] = np.log1p(images_count)
        
        # Признаки рекламы
        cpm = df.get('cpm', 0).fillna(0)
        proceeds = df.get('proceeds', 0).fillna(0)
        df['has_advertising'] = (cpm > 0).astype(int)
        df['ad_efficiency'] = safe_divide(proceeds, cpm)
        
        # Признаки потерь
        lost_proceeds = df.get('lost_proceeds', 0).fillna(0)
        df['loss_ratio'] = safe_divide(lost_proceeds, proceeds + lost_proceeds)
        df['revenue_potential'] = proceeds + lost_proceeds
        
        # Признаки наличия товара
        quantity = df.get('quantity', 0).fillna(0)
        df['stock_status'] = (quantity > 0).astype(int)
        
        # Создание категорий остатков с обработкой проблем
        try:
            df['stock_level'] = pd.cut(quantity, bins=4, labels=['out', 'low', 'medium', 'high'])
            df['stock_level'] = df['stock_level'].astype(str).fillna('out')
        except Exception:
            df['stock_level'] = 'out'
        
        # Промо активности
        in_promo = df.get('in_promo', False).fillna(False)
        df['promo_boost'] = in_promo.astype(int)
        
        # Финальная очистка всех созданных признаков
        new_features = [
            'is_discounted', 'price_per_rating', 'rating_weighted', 'delivery_position_ratio',
            'position_squared', 'position_log', 'discount_price_ratio', 'has_images',
            'image_quality_score', 'has_advertising', 'ad_efficiency', 'loss_ratio',
            'revenue_potential', 'stock_status', 'promo_boost'
        ]
        
        for feature in new_features:
            if feature in df.columns:
                df[feature] = df[feature].replace([np.inf, -np.inf], 0).fillna(0)
        
        print(f"Создано {len(new_features)} дополнительных признаков")
        return df
    
    def prepare_features_for_model(self, df):
        """
        Готовим финальный набор признаков для модели
        """
        # Числовые признаки
        numeric_features = [
            'price', 'discount', 'product_rating', 'seller_rating', 'reviews_count',
            'quantity', 'proceeds', 'lost_proceeds', 'images_count', 'cpm',
            'delivery_efficiency_wh_avg_pos', 'is_discounted', 'price_per_rating',
            'rating_weighted', 'delivery_position_ratio', 'position_squared',
            'position_log', 'discount_price_ratio', 'has_images', 'image_quality_score',
            'has_advertising', 'ad_efficiency', 'loss_ratio', 'revenue_potential',
            'stock_status', 'promo_boost'
        ]
        
        # Категориальные признаки для кодирования
        categorical_features = ['city_of_search', 'search_query', 'category', 'price_category', 'stock_level']
        
        # Подготовка данных
        feature_df = df.copy()
        
        # Кодирование категориальных признаков
        for cat_feature in categorical_features:
            if cat_feature in feature_df.columns:
                if cat_feature not in self.label_encoders:
                    self.label_encoders[cat_feature] = LabelEncoder()
                    feature_df[f'{cat_feature}_encoded'] = self.label_encoders[cat_feature].fit_transform(
                        feature_df[cat_feature].astype(str).fillna('unknown')
                    )
                else:
                    # Обработка новых значений при предсказании
                    try:
                        feature_df[f'{cat_feature}_encoded'] = self.label_encoders[cat_feature].transform(
                            feature_df[cat_feature].astype(str).fillna('unknown')
                        )
                    except ValueError:
                        # Если встретилось новое значение, заменяем на наиболее частое
                        most_frequent = feature_df[cat_feature].mode()[0] if not feature_df[cat_feature].empty else 'unknown'
                        feature_df[cat_feature] = feature_df[cat_feature].fillna(most_frequent)
                        feature_df[f'{cat_feature}_encoded'] = self.label_encoders[cat_feature].transform(
                            feature_df[cat_feature].astype(str)
                        )
                
                numeric_features.append(f'{cat_feature}_encoded')
        
        # Выбираем только доступные признаки
        available_features = [col for col in numeric_features if col in feature_df.columns]
        self.feature_columns = available_features
        
        return feature_df[available_features].fillna(0)
    
    def train_models(self, df, test_size=0.2):
        """
        Обучаем модели на данных с правильным разделением по артикулам
        """
        print(f"Начинаем с {len(df)} записей")
        
        # Подготовка данных
        processed_data = self.prepare_data_from_unified_dataset(df)
        X = self.prepare_features_for_model(processed_data)
        
        # Очистка данных от проблемных значений
        print("Убираем проблемные значения из данных...")
        
        # Заменяем бесконечные значения на NaN, затем на 0
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Проверяем наличие нужных колонок
        if 'position' not in processed_data.columns:
            raise ValueError("Колонка 'position' не найдена в данных")
        if 'proceeds' not in processed_data.columns:
            raise ValueError("Колонка 'proceeds' не найдена в данных")
        
        # Целевые переменные
        y_position_raw = processed_data['position'].copy()
        y_revenue_raw = processed_data['proceeds'].copy()
        
        # Очистка целевых переменных
        # Удаляем строки с плохими позициями или выручкой
        valid_position_mask = (y_position_raw > 0) & (~y_position_raw.isna()) & (y_position_raw != np.inf)
        valid_revenue_mask = (~y_revenue_raw.isna()) & (y_revenue_raw != np.inf) & (y_revenue_raw >= 0)
        
        # Объединяем условия
        valid_mask = valid_position_mask & valid_revenue_mask
        
        print(f"Осталось хороших записей: {valid_mask.sum()} из {len(df)}")
        
        if valid_mask.sum() < 10:
            raise ValueError(f"Слишком мало хороших записей для обучения: {valid_mask.sum()}")
        
        # Применяем фильтр
        X_clean = X[valid_mask].reset_index(drop=True)
        y_position_clean = y_position_raw[valid_mask].reset_index(drop=True)
        y_revenue_clean = y_revenue_raw[valid_mask].reset_index(drop=True)
        processed_data_clean = processed_data[valid_mask].reset_index(drop=True)
        
        # Логарифмическое преобразование позиции
        y_position = np.log1p(y_position_clean)
        y_revenue = y_revenue_clean
        
        # Финальная проверка на проблемные значения
        if X_clean.isna().any().any():
            print("Нашли пропуски в признаках, заполняем нулями...")
            X_clean = X_clean.fillna(0)
        
        if y_position.isna().any() or y_revenue.isna().any():
            raise ValueError("Нашли пропуски в целевых переменных после очистки")
        
        print(f"Финальный размер данных: {len(X_clean)} записей")
        print(f"Количество признаков: {X_clean.shape[1]}")
        
        # Правильное разделение данных по артикулам
        if 'product_id' in processed_data_clean.columns:
            print("Разделяем данные по артикулам (артикулы не пересекаются между выборками)...")
            
            # Получаем уникальные артикулы
            unique_articles = processed_data_clean['product_id'].unique()
            
            # Разделяем артикулы на train/test
            train_articles, test_articles = train_test_split(
                unique_articles, test_size=test_size, random_state=42
            )
            
            # Создаем маски для разделения данных
            train_mask = processed_data_clean['product_id'].isin(train_articles)
            test_mask = processed_data_clean['product_id'].isin(test_articles)
            
            # Разделяем данные
            X_train = X_clean[train_mask]
            X_test = X_clean[test_mask]
            y_pos_train = y_position[train_mask]
            y_pos_test = y_position[test_mask]
            y_rev_train = y_revenue[train_mask]
            y_rev_test = y_revenue[test_mask]
            
            print(f"Обучающая выборка: {len(X_train)} записей ({len(train_articles)} артикулов)")
            print(f"Тестовая выборка: {len(X_test)} записей ({len(test_articles)} артикулов)")
            
        else:
            print("Колонка 'product_id' не найдена, используем обычное разделение...")
            # Обычное разделение данных
            X_train, X_test, y_pos_train, y_pos_test = train_test_split(
                X_clean, y_position, test_size=test_size, random_state=42
            )
            _, _, y_rev_train, y_rev_test = train_test_split(
                X_clean, y_revenue, test_size=test_size, random_state=42
            )
        
        # Параметры LightGBM
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 100,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Обучение модели позиции
        print("Обучаем модель для предсказания позиции...")
        self.position_model = lgb.LGBMRegressor(**lgb_params)
        self.position_model.fit(X_train, y_pos_train)
        
        # Обучение модели выручки
        print("Обучаем модель для предсказания выручки...")
        self.revenue_model = lgb.LGBMRegressor(**lgb_params)
        self.revenue_model.fit(X_train, y_rev_train)
        
        # Оценка качества
        try:
            pos_pred = self.position_model.predict(X_test)
            rev_pred = self.revenue_model.predict(X_test)
            
            pos_r2 = r2_score(y_pos_test, pos_pred)
            rev_r2 = r2_score(y_rev_test, rev_pred)
            
            print(f"Качество модели позиции - R²: {pos_r2:.4f}")
            print(f"Качество модели выручки - R²: {rev_r2:.4f}")
        except Exception as e:
            print(f"Не удалось оценить качество: {e}")
            print("Модели обучены, но оценка качества недоступна")
        
        return self.position_model, self.revenue_model
    
    def predict(self, product_data):
        """
        Предсказываем позицию и выручку для нового товара
        """
        if self.position_model is None or self.revenue_model is None:
            raise ValueError("Модели не обучены!")
        
        # Подготовка данных
        processed_data = self.prepare_data_from_unified_dataset(product_data)
        X = self.prepare_features_for_model(processed_data)
        
        # Предсказания
        predicted_log_position = self.position_model.predict(X)[0]
        predicted_position = np.expm1(predicted_log_position)  # обратное преобразование
        predicted_revenue = self.revenue_model.predict(X)[0]
        
        return predicted_position, predicted_revenue
    
    def save_models(self, position_path='position_model.pkl', revenue_path='proceeds_model.pkl', features_path='feature_list.pkl'):
        """
        Сохраняем обученные модели
        """
        joblib.dump(self.position_model, position_path)
        joblib.dump(self.revenue_model, revenue_path)
        joblib.dump(self.feature_columns, features_path)
        print(f"Модели сохранены: {position_path}, {revenue_path}, {features_path}")
    
    def load_models(self, position_path='position_model.pkl', revenue_path='proceeds_model.pkl', features_path='feature_list.pkl'):
        """
        Загружаем обученные модели
        """
        self.position_model = joblib.load(position_path)
        self.revenue_model = joblib.load(revenue_path)
        self.feature_columns = joblib.load(features_path)
        print("Модели успешно загружены!")
    
    def analyze_feature_importance(self):
        """
        Анализируем важность признаков
        """
        if self.position_model is None or self.revenue_model is None:
            raise ValueError("Модели не обучены!")
        
        position_importance = dict(zip(self.feature_columns, self.position_model.feature_importances_))
        revenue_importance = dict(zip(self.feature_columns, self.revenue_model.feature_importances_))
        
        # Топ-10 наиболее важных признаков
        top_position_features = sorted(position_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_revenue_features = sorted(revenue_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("Топ-10 факторов для позиции:")
        for feature, importance in top_position_features:
            print(f"  {feature}: {importance:.4f}")
        
        print("\nТоп-10 факторов для выручки:")
        for feature, importance in top_revenue_features:
            print(f"  {feature}: {importance:.4f}")
        
        return position_importance, revenue_importance

# Функция для работы с основным кодом
def train_and_save_models(data_file_path):
    """
    Обучаем и сохраняем модели на основе датасета
    """
    # Загрузка данных
    print("Загружаем данные...")
    df = pd.read_csv(data_file_path)  # или любой другой формат данных
    
    # Инициализация модели
    model = WBRankingModel()
    
    # Обучение
    print("Начинаем обучение моделей...")
    model.train_models(df)
    
    # Анализ важности признаков
    print("\nАнализируем важность признаков:")
    model.analyze_feature_importance()
    
    # Сохранение моделей
    model.save_models()
    
    print("\nМодели успешно обучены и сохранены!")
    return model

if __name__ == "__main__":
    # Пример использования для обучения
    # model = train_and_save_models("your_dataset.csv")
    pass
