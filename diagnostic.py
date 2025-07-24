# diagnostic.py

# --- Копируем сюда ТОЛЬКО те функции, которые нам нужны для теста ---

def calculate_constructor_score(product_data):
    """Рассчитывает качество наполнения карточки товара (от 0 до 100)."""
    score = 0
    if isinstance(product_data.get("description", ""), str) and len(product_data.get("description", "")) > 500:
        score += 30
    if product_data.get("images_count", 0) >= 6:
        score += 30
    if len(str(product_data.get("product_name", "")).split()) >= 5:
        score += 25
    if product_data.get("category"):
        score += 15
    return min(100, score)


def calculate_ranking_factors_original(current_data):
    """
    ОРИГИНАЛЬНАЯ ВЕРСИЯ ВАШЕЙ ФУНКЦИИ.
    Мы используем ее, чтобы воспроизвести ошибку.
    """
    factors = {}
    discount_score = 50 if current_data.get("is_discounted") else 0
    discount_score += min(50, (current_data.get("discount", 0) / 30.0) * 50)
    factors["Цена и скидки"] = discount_score
    factors["Продажи и оборот"] = min(100, (current_data.get("proceeds", 0) / 250000.0) * 100)
    rating_score = (current_data.get("product_rating", 0) / 5.0) * 40
    rating_score += (current_data.get("seller_rating", 0) / 5.0) * 30
    rating_score += min(30, (current_data.get("reviews_count", 0) / 100.0) * 30)
    factors["Рейтинги и отзывы"] = rating_score
    factors["Качество карточки"] = calculate_constructor_score(current_data)
    # Вот вероятный источник проблемы:
    delivery_hours = current_data.get("delivery_efficiency_wh_avg_pos", 96)
    factors["Доставка"] = max(0, 100 - ((delivery_hours - 24) / 72.0) * 100)
    factors["Остатки на складе"] = min(100, (current_data.get("quantity", 0) / 500.0) * 100)
    return factors


# --- Основная часть диагностического скрипта ---

if __name__ == "__main__":
    print("--- ЗАПУСК ДИАГНОСТИКИ ФУНКЦИИ calculate_ranking_factors ---\n")

    # Создаем тестовый набор данных, который включает "опасные" значения
    test_case = {
        "description": "a" * 600,
        "images_count": 8,
        "product_name": "очень длинное название для теста",
        "category": "Тестовая категория",
        "is_discounted": True,
        "discount": 50,
        "proceeds": 300000,
        "product_rating": 5.0,
        "seller_rating": 5.0,
        "reviews_count": 200,
        # === Ключевой параметр для теста: очень быстрая доставка ===
        "delivery_efficiency_wh_avg_pos": 12, # Это значение < 24 и вызовет ошибку
        "quantity": 1000,
    }

    print("ИСПОЛЬЗУЕМ ТЕСТОВЫЕ ДАННЫЕ:")
    for key, value in test_case.items():
        print(f"  - {key}: {value}")
    print("-" * 20)

    # Вызываем вашу функцию с тестовыми данными
    calculated_factors = calculate_ranking_factors_original(test_case)

    print("\nРЕЗУЛЬТАТЫ РАСЧЕТА ФАКТОРОВ:")
    for factor, value in calculated_factors.items():
        print(f"  - {factor}: {value:.2f}")
    print("-" * 20)

    print("\nПРОВЕРКА ЗНАЧЕНИЙ НА КОРРЕКТНОСТЬ (ДИАПАЗОН 0-100):")
    has_error = False
    for factor, value in calculated_factors.items():
        if not (0 <= value <= 100):
            print(f"!!! НАЙДЕНА ПРОБЛЕМА !!!")
            print(f"Фактор '{factor}' имеет некорректное значение: {value:.2f}")
            print("Это значение вне диапазона [0, 100] и вызовет ошибку в st.progress().")
            has_error = True
        else:
            print(f"  - {factor}: {value:.2f} -> OK")

    if not has_error:
        print("\nВсе значения в порядке. Проблема может быть в другом месте.")
