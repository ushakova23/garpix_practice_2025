

import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class WBRankingModel:
    def __init__(self):
        self.position_model = None
        self.revenue_model = None
        self.label_encoders = {}
        self.feature_columns = None

    def get_feature_importance_dict(self):
        """Возвращает словарь с важностью признаков для модели позиции."""
        if self.position_model is None:
            raise ValueError("Модель позиции не обучена!")
        return dict(zip(self.feature_columns, self.position_model.feature_importances_))

    def prepare_data_from_unified_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ЕДИНЫЙ И НАДЕЖНЫЙ МЕТОД ДЛЯ ПОДГОТОВКИ ДАННЫХ И СОЗДАНИЯ ПРИЗНАКОВ.
        Работает как с одной строкой (для симуляции), так и с большим датафреймом (для обучения).
        """
        prepared_df = df.copy()

        # --- Этап 1: Гарантируем наличие и очистку ВСЕХ исходных колонок ---
        # Это ключевое исправление. Мы сначала приводим данные в порядок, а потом создаем признаки.
        
        all_numeric_cols = [
            "discount", "product_rating", "price", "proceeds", "seller_rating",
            "reviews_count", "quantity", "images_count", "lost_proceeds",
            "delivery_efficiency_wh_avg_pos", "cpm"
        ]
        for col in all_numeric_cols:
            if col not in prepared_df.columns:
                prepared_df[col] = 0 # Создаем колонку, если ее нет
            # Преобразуем в число и заполняем пропуски
            prepared_df[col] = pd.to_numeric(prepared_df[col], errors='coerce').fillna(0)

        all_categorical_cols = ["category", "search_query", "city_of_search", "in_promo"]
        for col in all_categorical_cols:
            if col not in prepared_df.columns:
                prepared_df[col] = False if col == 'in_promo' else "unknown"
            prepared_df[col] = prepared_df[col].fillna(False if col == 'in_promo' else "unknown")

        # --- Этап 2: Создание новых признаков из очищенных данных ---
        
        def safe_divide(numerator, denominator, default=0):
            """Безопасное деление, которое теперь работает с очищенными Series."""
            # Добавляем малое число для избежания деления на 0
            result = numerator / (denominator + 1e-6)
            return result.replace([np.inf, -np.inf], default).fillna(default)

        # Теперь мы можем безопасно обращаться к колонкам, не используя .get() или .fillna()
        prepared_df["is_discounted"] = (prepared_df["discount"] > 0).astype(int)
        prepared_df["price_per_rating"] = safe_divide(prepared_df["price"], prepared_df["product_rating"])
        prepared_df["rating_weighted"] = prepared_df["product_rating"] * np.log1p(prepared_df["reviews_count"])
        prepared_df["discount_ratio"] = safe_divide(prepared_df["discount"], prepared_df["price"])
        prepared_df["has_images"] = (prepared_df["images_count"] > 0).astype(int)
        prepared_df["image_quality_score"] = np.log1p(prepared_df["images_count"])
        prepared_df["has_advertising"] = (prepared_df["cpm"] > 0).astype(int)
        prepared_df["ad_efficiency"] = safe_divide(prepared_df["proceeds"], prepared_df["cpm"])
        prepared_df["loss_ratio"] = safe_divide(prepared_df["lost_proceeds"], prepared_df["proceeds"] + prepared_df["lost_proceeds"])
        prepared_df["revenue_potential"] = prepared_df["proceeds"] + prepared_df["lost_proceeds"]
        prepared_df["stock_status"] = (prepared_df["quantity"] > 0).astype(int)
        prepared_df["promo_boost"] = prepared_df["in_promo"].astype(int)
        prepared_df['seller_vs_product_rating_diff'] = prepared_df["seller_rating"] - prepared_df["product_rating"]

        # Категоризация с защитой от ошибок на единичных значениях
        try:
            prepared_df["price_category"] = pd.cut(prepared_df["price"], bins=5, labels=["low", "below_avg", "avg", "above_avg", "high"])
        except (ValueError, TypeError):
            prepared_df["price_category"] = "unknown"
        prepared_df["price_category"] = prepared_df["price_category"].astype(str).fillna("unknown")

        try:
            prepared_df["stock_level"] = pd.cut(prepared_df["quantity"], bins=4, labels=["out", "low", "medium", "high"])
        except (ValueError, TypeError):
            prepared_df["stock_level"] = "out"
        prepared_df["stock_level"] = prepared_df["stock_level"].astype(str).fillna("out")
        
        return prepared_df

    def prepare_features_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Финальная подготовка признаков для модели, включая кодирование."""
        feature_df = df.copy()

        numeric_features = [
            "price", "discount", "product_rating", "seller_rating", "reviews_count", "quantity",
            "lost_proceeds", "images_count", "cpm", "delivery_efficiency_wh_avg_pos",
            "is_discounted", "price_per_rating", "rating_weighted", "discount_ratio", "has_images",
            "image_quality_score", "has_advertising", "ad_efficiency", "loss_ratio",
            "revenue_potential", "stock_status", "promo_boost", "seller_vs_product_rating_diff"
        ]
        
        categorical_features = ["city_of_search", "search_query", "category", "price_category", "stock_level"]
        
        final_feature_list = list(numeric_features)

        for cat_feature in categorical_features:
            encoded_col_name = f"{cat_feature}_encoded"
            feature_df[cat_feature] = feature_df[cat_feature].astype(str).fillna("unknown")

            if cat_feature in self.label_encoders:
                le = self.label_encoders[cat_feature]
                new_classes = set(feature_df[cat_feature]) - set(le.classes_)
                if new_classes:
                    le.classes_ = np.array(list(le.classes_) + list(new_classes))
                
                # Применяем transform, заменяя неизвестные на 'unknown' (если вдруг пропустили)
                known_mask = feature_df[cat_feature].isin(le.classes_)
                feature_df.loc[~known_mask, cat_feature] = 'unknown'
                feature_df[encoded_col_name] = le.transform(feature_df[cat_feature])
            else:
                le = LabelEncoder()
                # Учим на всех данных, включая 'unknown'
                feature_df[encoded_col_name] = le.fit_transform(feature_df[cat_feature])
                self.label_encoders[cat_feature] = le
            
            final_feature_list.append(encoded_col_name)
        
        # Если мы в режиме обучения, сохраняем список признаков
        if self.feature_columns is None:
            self.feature_columns = final_feature_list
        
        # Убедимся, что все колонки на месте и в правильном порядке
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0 # Добавляем недостающие колонки

        return feature_df[self.feature_columns].fillna(0)

    # ... Остальные методы (train_models, predict, save_models, и т.д.) остаются без изменений ...

    def train_models(self, df, test_size=0.2):
        print(f"Начинаем с {len(df)} записей")
        processed_data = self.prepare_data_from_unified_dataset(df)
        X = self.prepare_features_for_model(processed_data)
        
        y_position_raw = processed_data["position"]
        y_revenue_raw = processed_data["proceeds"]

        valid_mask = (y_position_raw > 0) & (~y_position_raw.isna()) & (y_revenue_raw >= 0) & (~y_revenue_raw.isna())
        print(f"Осталось хороших записей: {valid_mask.sum()} из {len(df)}")
        if valid_mask.sum() < 50:
            raise ValueError(f"Слишком мало хороших записей для обучения: {valid_mask.sum()}")

        X_clean = X.loc[valid_mask].reset_index(drop=True)
        y_position_clean = y_position_raw.loc[valid_mask].reset_index(drop=True)
        y_revenue_clean = y_revenue_raw.loc[valid_mask].reset_index(drop=True)
        
        y_position = np.log1p(y_position_clean)
        y_revenue = y_revenue_clean

        print(f"Финальный размер данных: {len(X_clean)} записей, {X_clean.shape[1]} признаков")
        
        X_train, X_test, y_pos_train, y_pos_test, y_rev_train, y_rev_test = train_test_split(
            X_clean, y_position, y_revenue, test_size=test_size, random_state=42
        )

        lgb_params = {"objective": "regression", "metric": "rmse", "boosting_type": "gbdt", "num_leaves": 120,
                      "learning_rate": 0.03, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                      "verbose": -1, "random_state": 42, "n_estimators": 1000}

        print("Обучаем модель для предсказания позиции...")
        self.position_model = lgb.LGBMRegressor(**lgb_params)
        self.position_model.fit(X_train, y_pos_train, eval_set=[(X_test, y_pos_test)],
                                callbacks=[lgb.early_stopping(100, verbose=False)])

        print("Обучаем модель для предсказания выручки...")
        self.revenue_model = lgb.LGBMRegressor(**lgb_params)
        self.revenue_model.fit(X_train, y_rev_train, eval_set=[(X_test, y_rev_test)],
                               callbacks=[lgb.early_stopping(100, verbose=False)])

        pos_r2 = r2_score(y_pos_test, self.position_model.predict(X_test))
        rev_r2 = r2_score(y_rev_test, self.revenue_model.predict(X_test))
        print(f"Качество модели позиции - R²: {pos_r2:.4f}")
        print(f"Качество модели выручки - R²: {rev_r2:.4f}")
        return self.position_model, self.revenue_model

    def predict(self, product_data):
        if not all([self.position_model, self.revenue_model, self.feature_columns, self.label_encoders is not None]):
            raise ValueError("Модели или их компоненты не загружены!")
        
        if isinstance(product_data, dict):
            product_data = pd.DataFrame([product_data])

        processed_data = self.prepare_data_from_unified_dataset(product_data)
        X = self.prepare_features_for_model(processed_data)

        predicted_log_position = self.position_model.predict(X)[0]
        predicted_position = np.expm1(predicted_log_position)
        predicted_revenue = self.revenue_model.predict(X)[0]

        return predicted_position, predicted_revenue

    def save_models(self, position_path="models/position_model.pkl", revenue_path="models/proceeds_model.pkl",
                    features_path="models/feature_list.pkl", encoders_path="models/label_encoders.pkl"):
        joblib.dump(self.position_model, position_path)
        joblib.dump(self.revenue_model, revenue_path)
        joblib.dump(self.feature_columns, features_path)
        joblib.dump(self.label_encoders, encoders_path)
        print(f"Модели сохранены в папку 'models/'")

    def load_models(self, position_path="models/position_model.pkl", revenue_path="models/proceeds_model.pkl",
                    features_path="models/feature_list.pkl", encoders_path="models/label_encoders.pkl"):
        self.position_model = joblib.load(position_path)
        self.revenue_model = joblib.load(revenue_path)
        self.feature_columns = joblib.load(features_path)
        self.label_encoders = joblib.load(encoders_path)
        print("Модели успешно загружены!")
    

    def analyze_feature_importance(self):
        """
        Анализируем важность признаков
        """
        if self.position_model is None or self.revenue_model is None:
            raise ValueError("Модели не обучены!")

        position_importance = dict(
            zip(self.feature_columns, self.position_model.feature_importances_)
        )
        revenue_importance = dict(
            zip(self.feature_columns, self.revenue_model.feature_importances_)
        )

        # Топ-10 наиболее важных признаков
        top_position_features = sorted(
            position_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        top_revenue_features = sorted(
            revenue_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]

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
