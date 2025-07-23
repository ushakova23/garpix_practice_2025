from unittest.mock import Mock, patch

import pytest


# Мокаем streamlit
@pytest.fixture(autouse=True)
def mock_streamlit():
    with patch.multiple(
        "streamlit",
        title=Mock(),
        markdown=Mock(),
        text_input=Mock(return_value="test"),
        selectbox=Mock(return_value="Москва"),
        session_state={},
    ):
        yield


class TestStreamlitApp:

    def test_prepare_model_input(self):
        """Тест подготовки данных для модели"""
        from src.streamlit_app_v2 import prepare_model_input

        product_context = {
            "id": 12345,
            "expected_position": 5,
            "wh_avg_position": 15.5,
            "brand": {"id": 123},
        }

        product_details = {
            "price": 1200,
            "rating": 4.5,
            "proceeds": 15000,
            "images": ["img1.jpg", "img2.jpg"],
        }

        brand_details = {"rating": 4.5, "reviews": 1000}

        result = prepare_model_input(
            product_context, product_details, brand_details, "Москва", "футболка"
        )

        assert isinstance(result, dict)
        assert result["product_id"] == 12345
        assert result["city_of_search"] == "Москва"
        print("✅ Подготовка данных для модели работает")

    def test_calculate_ranking_factors(self):
        """Тест расчета факторов ранжирования"""
        from src.streamlit_app_v2 import calculate_ranking_factors

        current_data = {
            "is_discounted": 1,
            "discount": 15,
            "proceeds": 10000,
            "seller_rating": 4.5,
            "product_rating": 4.2,
        }

        factors = calculate_ranking_factors(current_data)

        assert isinstance(factors, dict)
        if "Акции" in factors:
            assert factors["Акции"] == 15
        print("✅ Расчет факторов ранжирования работает")
