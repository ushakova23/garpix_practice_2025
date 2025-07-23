from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.wb_ranking_model import WBRankingModel


class TestWBRankingModel:

    @pytest.fixture
    def model(self):
        return WBRankingModel()

    @pytest.fixture
    def sample_data(self):
        # Добавляем ВСЕ нужные колонки
        return pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5],
                "position": [1, 5, 10, 15, 20],
                "proceeds": [10000, 5000, 3000, 2000, 1000],
                "price": [1000, 800, 600, 400, 200],
                "discount": [10, 0, 15, 5, 0],
                "product_rating": [4.5, 4.0, 3.5, 3.0, 2.5],
                "seller_rating": [4.8, 4.2, 3.8, 3.2, 2.8],
                "reviews_count": [100, 50, 30, 20, 10],
                "quantity": [50, 30, 20, 10, 5],
                "lost_proceeds": [1000, 500, 300, 200, 100],
                "images_count": [8, 6, 4, 3, 2],
                "cpm": [50, 30, 20, 10, 5],  # ДОБАВИЛИ
                "delivery_efficiency_wh_avg_pos": [10, 20, 30, 40, 50],  # ДОБАВИЛИ
                "in_promo": [True, False, True, False, True],  # ДОБАВИЛИ
                "city_of_search": ["Москва", "СПб", "Москва", "СПб", "Москва"],
                "search_query": [
                    "футболка",
                    "джинсы",
                    "футболка",
                    "джинсы",
                    "футболка",
                ],
                "category": ["Одежда", "Одежда", "Одежда", "Одежда", "Одежда"],
            }
        )

    def test_init(self, model):
        """Тест инициализации модели"""
        assert model.position_model is None
        assert model.revenue_model is None
        assert model.label_encoders == {}
        print("✅ Инициализация работает")

    def test_create_features(self, model, sample_data):
        """Тест создания признаков"""
        result = model._create_features_from_existing_data(sample_data)

        # Проверяем новые признаки
        assert "is_discounted" in result.columns
        assert "price_per_rating" in result.columns
        assert len(result) == len(sample_data)
        print("✅ Создание признаков работает")

    def test_prepare_features(self, model, sample_data):
        """Тест подготовки признаков для модели"""
        processed_data = model._create_features_from_existing_data(sample_data)
        features = model.prepare_features_for_model(processed_data)

        assert isinstance(features, pd.DataFrame)
        assert not features.isna().any().any()  # Нет пропусков
        print("✅ Подготовка признаков работает")

    @patch("joblib.dump")
    def test_save_models(self, mock_dump, model):
        """Тест сохранения моделей"""
        model.position_model = Mock()
        model.revenue_model = Mock()
        model.feature_columns = ["feature1", "feature2"]

        model.save_models()
        assert mock_dump.call_count == 3
        print("✅ Сохранение моделей работает")
