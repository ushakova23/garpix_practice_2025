from unittest.mock import Mock, patch

import pytest
import requests

from src.wildbox_client import get_brand_details, get_product_details


class TestWildboxClient:

    @patch("src.wildbox_client.requests.get")
    def test_get_product_success(self, mock_get):
        """Тест успешного получения данных товара"""
        # Мокаем успешный ответ
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [{"id": 12345, "price": 1200, "rating": 4.5}]
        }
        mock_get.return_value = mock_response

        result = get_product_details(12345)

        assert result == {"id": 12345, "price": 1200, "rating": 4.5}
        assert mock_get.called
        print("✅ Получение данных товара работает")

    @patch("src.wildbox_client.requests.get")
    def test_get_product_error(self, mock_get):
        """Тест обработки ошибки API"""
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        result = get_product_details(12345)
        assert result == {}
        print("✅ Обработка ошибок API работает")

    @patch("src.wildbox_client.requests.get")
    def test_get_brand_success(self, mock_get):
        """Тест получения данных бренда"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "results": [{"rating": 4.5, "reviews": 1000}]
        }
        mock_get.return_value = mock_response

        result = get_brand_details(123)

        assert result == {"rating": 4.5, "reviews": 1000}
        print("✅ Получение данных бренда работает")
