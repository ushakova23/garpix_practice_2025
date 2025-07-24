import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.wildbox_client import (get_brand_details, get_product_details,
                                get_warehouse_positions)


def prepare_model_input(
    product_context, product_details, brand_details, city, search_query
):
    """Собирает все данные в один словарь, готовый для модели."""
    subject_info = product_details.get("subject", {})
    category_name = subject_info.get("name") if isinstance(subject_info, dict) else None
    brand_id = product_context.get("brand", {}).get("id")
    raw_wh_avg_pos = product_context.get("wh_avg_position")
    try:
        wh_avg_pos_numeric = float(raw_wh_avg_pos)
    except (ValueError, TypeError):
        wh_avg_pos_numeric = 0.0

    data_dict = {
        "product_id": product_context.get("id"),
        "brand_id": brand_id,
        "category": category_name,
        "search_query": search_query,
        "city_of_search": city,
        "position": product_context.get("expected_position"),
        "proceeds": product_details.get("proceeds"),
        "delivery_efficiency_wh_avg_pos": wh_avg_pos_numeric,
        "in_promo": bool(product_details.get("promos")),
        "price": product_details.get("price"),
        "discount": product_details.get("discount"),
        "product_rating": product_details.get("rating"),
        "seller_rating": brand_details.get("rating"),
        "reviews_count": brand_details.get("reviews"),
        "quantity": product_details.get("quantity"),
        "lost_proceeds": product_details.get("lost_proceeds"),
        "images_count": len(product_details.get("images", [])),
        "product_name": product_details.get("name", ""),
        "description": product_details.get("description", ""),
        "cpm": product_details.get("cpm", 0)
    }
    temp_df = pd.DataFrame([data_dict])

    numeric_cols = temp_df.select_dtypes(include=np.number).columns
    temp_df[numeric_cols] = temp_df[numeric_cols].fillna(0)

    temp_df["is_discounted"] = (temp_df["discount"] > 0).astype(int)
    temp_df["price_per_rating"] = (
        (temp_df["price"] / temp_df["product_rating"])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    return temp_df.iloc[0].to_dict()


def calculate_constructor_score(product_data):
    """Рассчитывает качество наполнения карточки товара (от 0 до 100)."""
    if not product_data:
        return 0
    
    score = 0
    description = product_data.get("description", "")
    if isinstance(description, str) and len(description) > 500:
        score += 30
    
    if product_data.get("images_count", 0) >= 6:
        score += 30
    
    product_name = product_data.get("product_name", "")
    if len(str(product_name).split()) >= 5:
        score += 25
    
    if product_data.get("category"):
        score += 15
    
    return min(100, score)


def calculate_ranking_factors(current_data):
    """
    Рассчитывает оценку ключевых факторов ранжирования для товара (от 0 до 100).
    """
    if not current_data:
        return {factor: 0 for factor in ["Цена и скидки", "Продажи и оборот", "Рейтинги и отзывы",
                                        "Качество карточки", "Доставка", "Остатки на складе"]}
    
    factors = {}
    
    # Цена и Скидки
    discount_score = 50 if current_data.get("is_discounted", 0) else 0
    discount_score += min(50, (current_data.get("discount", 0) / 30.0) * 50)
    factors["Цена и скидки"] = discount_score

    # Продажи и оборот (шкала до 250,000 руб/мес)
    factors["Продажи и оборот"] = min(100, (current_data.get("proceeds", 0) / 250000.0) * 100)

    # Рейтинги и отзывы
    rating_score = (current_data.get("product_rating", 0) / 5.0) * 40
    rating_score += (current_data.get("seller_rating", 0) / 5.0) * 30
    rating_score += min(30, (current_data.get("reviews_count", 0) / 100.0) * 30)
    factors["Рейтинги и отзывы"] = rating_score

    # Качество карточки
    factors["Качество карточки"] = calculate_constructor_score(current_data)

    # Доставка (шкала от 24 до 96 часов)
    delivery_hours = current_data.get("delivery_efficiency_wh_avg_pos", 96)
    delivery_score = 100 - ((delivery_hours - 24) / 72.0) * 100
    factors["Доставка"] = max(0, delivery_score)
    
    # Остатки на складе (шкала до 500 шт)
    factors["Остатки на складе"] = min(100, (current_data.get("quantity", 0) / 500.0) * 100)

    # Финальная проверка
    for key in factors:
        factors[key] = max(0, min(100, factors[key]))

    return factors


def generate_optimization_recommendations(current_data):
    """Генерирует умные рекомендации, всегда находя точки роста."""
    if not current_data:
        return [{"factor": "Данные недоступны", "improvement": "Проверьте подключение к API",
                "position_change": "-", "revenue_change": "-", "cost": "-"}]
    
    recommendations = []
    
    if current_data.get("discount", 0) < 15 and current_data.get("price", 0) > 0:
        recommendations.append({
            "factor": "Усилить акцию",
            "improvement": "Увеличить скидку до 20-25%",
            "position_change": "Значительный буст",
            "revenue_change": "Привлечение покупателей",
            "cost": f"~{int(current_data['price'] * 0.1):,}₽ доп. с единицы"
        })

    if calculate_constructor_score(current_data) < 90:
        recommendations.append({
            "factor": "Улучшить карточку",
            "improvement": "Добавить фото/видео, расширить описание",
            "position_change": "Рост CTR",
            "revenue_change": "Больше доверия и продаж",
            "cost": "от 5,000₽ (контент)"
        })

    if current_data.get("product_rating", 0) < 4.8:
        recommendations.append({
            "factor": "Поднять рейтинг",
            "improvement": f"С {current_data.get('product_rating', 0):.1f} до 4.8+ через работу с отзывами",
            "position_change": "Выше в поиске",
            "revenue_change": "Повышение лояльности",
            "cost": "от 10,000₽ (сервисы/качество)"
        })

    if current_data.get("proceeds", 0) < 50000:
        recommendations.append({
            "factor": "Увеличить продажи",
            "improvement": "Запустить внутреннюю рекламу (поиск/каталог)",
            "position_change": "Резкий рост видимости",
            "revenue_change": "Наращивание оборота",
            "cost": "от 15,000₽ (бюджет)"
        })
    
    if current_data.get("delivery_efficiency_wh_avg_pos", 100) > 48:
         recommendations.append({
             "factor": "Ускорить доставку",
             "improvement": "Распределить товар по региональным складам",
             "position_change": "Приоритет в выдаче",
             "revenue_change": "Рост заказов из регионов",
             "cost": "Логистические расходы"
         })

    if not recommendations:
        recommendations.append({
            "factor": "Анализ конкурентов",
            "improvement": "Найти слабые места у топ-5 и превзойти их",
            "position_change": "Стратегическое преимущество",
            "revenue_change": "Отстройка от рынка",
            "cost": "Время на анализ"
        })

    return recommendations[:4]


def get_full_report(city: str, search_query: str, product_id: int):
    """Главная функция"""
    try:
        model_pos = joblib.load("models/position_model.pkl")
        model_proc = joblib.load("models/proceeds_model.pkl")
        features = joblib.load("models/feature_list.pkl")
    except FileNotFoundError:
        return None, "Файлы моделей не найдены.", None, None

    try:
        product_details = get_product_details(product_id)
        essential_fields = ["id", "price", "rating"]
        has_essential_data = any(field in product_details for field in essential_fields)

        if not product_details or not has_essential_data:
            return (
                None,
                f"Товар с артикулом {product_id} не найден или данные неполные.",
                None,
                None,
            )

    except Exception as e:
        return (
            None,
            f"Ошибка API при получении данных товара {product_id}: {e}",
            None,
            None,
        )

    brand_id = (
        product_details.get("brand", {}).get("id")
        if isinstance(product_details.get("brand"), dict)
        else None
    )

    try:
        brand_details = get_brand_details(brand_id) if brand_id else {}
    except Exception:
        brand_details = {}

    try:
        positions_data = get_warehouse_positions(product_id, search_query)
        current_position = None
        if positions_data:
            city_position_data = next(
                (p for p in positions_data if p.get("city") == city), None
            )
            if city_position_data:
                current_position = city_position_data.get("position")
    except Exception:
        current_position = None

    product_context = {
        "id": product_id,
        "expected_position": current_position,
        "position": current_position,
        "wh_avg_position": product_details.get("wh_avg_position"),
        "brand": {"id": brand_id},
    }

    try:
        model_input_data = prepare_model_input(
            product_context, product_details, brand_details, city, search_query
        )
    except Exception as e:
        return None, f"Ошибка подготовки данных для модели: {e}", None, None

    if model_input_data["position"] is None:
        model_input_data["position"] = 9999

    try:
        product_df = pd.DataFrame([model_input_data])

        missing_features = [f for f in features if f not in product_df.columns]
        for feature in missing_features:
            product_df[feature] = 0

        for col in ["city_of_search", "search_query", "category"]:
            if col in product_df.columns:
                product_df[col] = product_df[col].astype("category")

        base_position = np.expm1(model_pos.predict(product_df[features])[0])
        base_proceeds = model_proc.predict(product_df[features])[0]

    except Exception as e:
        return None, f"Ошибка работы модели: {e}", None, None

    model_input_data["position_real"] = current_position

    return model_input_data, base_position, base_proceeds, None


# --- UI часть приложения Streamlit ---

st.set_page_config(layout="wide", page_title="AI-Аналитик WB", page_icon="🤖")

# Заголовок
st.title("🤖 Модель оптимизации рекламных кампаний Wildberries")
st.markdown(
    "**Анализ факторов ранжирования и автоматическая оптимизация для увеличения выручки**"
)

# Поля для ввода
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        product_id_input = st.text_input(
            "Артикул товара (ID)", value="32395071", placeholder="Например: 16443420"
        )
    with col2:
        search_query_input = st.text_input(
            "Поисковый запрос",
            value="футболка женская белая",
            placeholder="Например: футболка женская белая",
        )
    with col3:
        city_input = st.selectbox(
            "Город",
            [
                "Москва",
                "Санкт-Петербург",
                "Новосибирск",
                "Хабаровск",
                "Екатеринбург",
                "Казань",
                "Краснодар",
            ],
        )

# Кнопка для запуска анализа
if st.button("🚀 Получить рекомендации", type="primary", use_container_width=True):
    if not product_id_input or not product_id_input.strip():
        st.error("Пожалуйста, введите артикул товара.")
    elif not product_id_input.isdigit():
        st.error("Артикул должен содержать только цифры.")
    else:
        with st.spinner("Анализирую данные с Wildbox и прогоняю через AI-модели..."):
            current_data, base_pos, base_proc, _ = get_full_report(
                city_input, search_query_input, int(product_id_input)
            )

        # Сохраняем результаты в session_state
        st.session_state["analysis_complete"] = True
        st.session_state["current_data"] = current_data
        st.session_state["base_pos"] = base_pos
        st.session_state["base_proc"] = base_proc

        # Сохраняем модели для симулятора
        try:
            model_pos = joblib.load("models/position_model.pkl")
            model_proc = joblib.load("models/proceeds_model.pkl")
            features = joblib.load("models/feature_list.pkl")
            st.session_state["model_pos"] = model_pos
            st.session_state["model_proc"] = model_proc
            st.session_state["features"] = features
        except:
            pass  # Модели недоступны

# Проверяем, есть ли сохраненные результаты анализа
if st.session_state.get("analysis_complete") and st.session_state.get("current_data"):
    current_data = st.session_state["current_data"]
    base_pos = st.session_state["base_pos"]
    base_proc = st.session_state["base_proc"]

    if not current_data:
        st.error(base_pos)
    else:
        # === ОСНОВНЫЕ МЕТРИКИ ===
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            current_pos = current_data.get("position_real")
            pos_display = str(current_pos) if current_pos is not None else "Не найден"
            st.metric(label="🎯 Текущая позиция", value=pos_display, delta=None)

        with col2:
            # Только если есть реальные данные для расчета
            factors_count = len(calculate_ranking_factors(current_data))
            if factors_count > 0:
                st.metric(
                    label="📊 Доступно факторов",
                    value=f"{factors_count} из 6",
                    delta=None,
                )
            else:
                st.metric(label="📊 Данные", value="Недоступны", delta=None)

        with col3:
            if base_proc:
                st.metric(
                    label="💰 Прогноз выручки", value=f"{base_proc:,.0f}₽", delta=None
                )
            else:
                st.metric(label="💰 Выручка", value="Нет данных", delta=None)

        with col4:
            price = current_data.get("price")
            if price:
                st.metric(
                    label="💵 Цена товара",
                    value=f"{price:,.0f}₽",
                    delta=(
                        f"-{current_data.get('discount', 0)}%"
                        if current_data.get("discount")
                        else None
                    ),
                )
            else:
                st.metric(label="💵 Цена", value="Нет данных", delta=None)

        st.divider()

        # === ФАКТОРЫ РАНЖИРОВАНИЯ (из первого кода) ===
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📊 Влияние факторов на позицию")
            st.caption("Оценка AI-модели на основе ваших данных")
            try:
                if os.path.exists("models/feature_importance.json"):
                    with open("models/feature_importance.json", "r", encoding="utf-8") as f:
                        tech_importances = json.load(f)
                    
                    factor_groups = {
                        "Продажи и оборот": ["proceeds"],
                        "Цена и скидки": ["price", "discount"],
                        "Доставка": ["delivery"],
                        "Рейтинги и отзывы": ["rating", "reviews"],
                        "Качество карточки": ["images", "category", "query"],
                        "Остатки на складе": ["quantity", "stock"],
                        "Реклама": ["cpm", "ad_"]
                    }
                    
                    agg_importance = {
                        group: sum(imp for feat, imp in tech_importances.items() if any(k in feat for k in keys))
                        for group, keys in factor_groups.items()
                    }
                    
                    total_sum = sum(agg_importance.values())
                    if total_sum > 0:
                        df = pd.DataFrame([
                            {"Фактор": k, "Важность (%)": v / total_sum * 100}
                            for k, v in agg_importance.items() if v > 0
                        ]).sort_values("Важность (%)")
                        
                        fig = px.bar(df, x='Важность (%)', y='Фактор', orientation="h",
                                   color='Важность (%)', color_continuous_scale=px.colors.sequential.Blues,
                                   text='Важность (%)')
                        fig.update_layout(height=400, showlegend=False, xaxis_title=None, yaxis_title=None,
                                        margin=dict(l=0, r=0, t=0, b=0),
                                        xaxis_range=[0, df['Важность (%)'].max() * 1.1])
                        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Данные о важности факторов недоступны")
                else:
                    st.info("Файл с важностью факторов не найден")
            except Exception as e:
                st.warning(f"Не удалось построить график важности факторов: {e}")

        with col2:
            st.subheader("🎯 Оценка вашего товара")
            st.caption("Насколько хорошо реализован каждый фактор (0-100%)")
            factors_data = calculate_ranking_factors(current_data)
            display_factors = ["Продажи и оборот", "Цена и скидки", "Доставка", "Рейтинги и отзывы", "Качество карточки", "Остатки на складе"]
            
            for factor in display_factors:
                value_int = int(factors_data.get(factor, 0))
                
                color = "🟢" if value_int >= 75 else "🟡" if value_int >= 40 else "🔴"
                st.markdown(f"{color} **{factor}**")
                
                st.progress(value_int / 100)
                
                st.markdown(
                    f"<div style='text-align: right; margin-top: -30px; margin-bottom: 10px; font-weight: 500;'><small>{value_int}%</small></div>",
                    unsafe_allow_html=True
                )

        st.divider()

        # === РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ (из первого кода) ===
        st.subheader("💡 Топ рекомендации по оптимизации")
        recommendations = generate_optimization_recommendations(current_data)
        rec_df = pd.DataFrame(recommendations).rename(columns={
            "factor": "ДЕЙСТВИЕ",
            "improvement": "ИЗМЕНЕНИЕ",
            "position_change": "ЭФФЕКТ НА ПОЗИЦИИ",
            "revenue_change": "ЭФФЕКТ НА ПРОДАЖИ",
            "cost": "СТОИМОСТЬ"
        })
        st.dataframe(rec_df, use_container_width=True, hide_index=True)

        st.divider()

        # === СИМУЛЯТОР ИЗМЕНЕНИЙ ===
        st.subheader("🔧 Симулятор изменений параметров")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("**Настройте параметры для симуляции:**")

            factors_data = calculate_ranking_factors(current_data)

            promo_participation = st.slider(
                "Участие в акциях:",
                0,
                100,
                max(20, int(factors_data.get("Цена и скидки", 20))),
                key="promo_slider",
            )
            sales_volume = st.slider(
                "Объем продаж:",
                0,
                100,
                max(45, int(factors_data.get("Продажи и оборот", 45))),
                key="sales_slider",
            )
            product_rating = st.slider(
                "Рейтинг товара:",
                0,
                100,
                max(75, int(factors_data.get("Рейтинги и отзывы", 75))),
                key="rating_slider",
            )
            card_completion = st.slider(
                "Наполнение карточки:",
                0,
                100,
                max(70, int(factors_data.get("Качество карточки", 70))),
                key="card_slider"
            )
            delivery_quality = st.slider(
                "Доставка:",
                0,
                100,
                max(60, int(factors_data.get("Доставка", 60))),
                key="delivery_slider",
            )
            price = st.slider(
                "Цена товара:",
                500,
                5000,
                max(1200, int(current_data.get("price", 1200))),
                key="price_slider",
            )

            # Дополнительные опции
            constructor_option = st.checkbox(
                "Опция конструктора (+3%)", key="constructor_checkbox"
            )
            wb_club_discount = st.checkbox("Скидка WB Клуба (+2%)", key="club_checkbox")

        with col2:
            st.markdown("**Влияние на метрики:**")

            # Расчет изменений на основе текущих значений слайдеров
            position_change = 0
            revenue_change = 0

            try:
                # Влияние акций (более чувствительное)
                base_promo = factors_data.get("Цена и скидки", 20)
                promo_diff = (promo_participation - base_promo) / 10.0  # Уменьшили делитель
                if abs(promo_diff) > 0.1:  # Реагируем на малые изменения
                    position_change -= promo_diff * 5  # Увеличили множитель
                    revenue_change += promo_diff * 6000

                # Влияние продаж (новый фактор)
                base_sales = factors_data.get("Продажи и оборот", 45)
                sales_diff = (sales_volume - base_sales) / 15.0
                if abs(sales_diff) > 0.1:
                    position_change -= sales_diff * 8  # Продажи сильно влияют на позицию
                    revenue_change += sales_diff * 8000

                # Влияние рейтинга (более чувствительное)
                base_rating = factors_data.get("Рейтинги и отзывы", 75)
                rating_diff = (product_rating - base_rating) / 15.0  # Уменьшили делитель
                if abs(rating_diff) > 0.1:
                    position_change -= rating_diff * 6  # Увеличили множитель
                    revenue_change += rating_diff * 4000

                # Влияние карточки (скорректированное)
                base_card = factors_data.get("Качество карточки", 70)
                card_diff = (card_completion - base_card) / 12.0  # Уменьшили делитель
                if abs(card_diff) > 0.1:
                    position_change -= card_diff * 4
                    revenue_change += card_diff * 3500

                # Влияние доставки (более заметное)
                base_delivery = factors_data.get("Доставка", 60)
                delivery_diff = (delivery_quality - base_delivery) / 12.0
                if abs(delivery_diff) > 0.1:
                    position_change -= delivery_diff * 7  # Доставка очень важна
                    revenue_change += delivery_diff * 5000

                # Влияние цены (обратная зависимость)
                current_price = current_data.get("price", 1200)
                price_change_pct = (price - current_price) / current_price
                if abs(price_change_pct) > 0.05:  # Реагируем на изменения > 5%
                    position_change += price_change_pct * 15  # Выше цена = хуже позиция
                    revenue_change -= price_change_pct * 3000  # Но может компенсироваться объемом

                # Влияние конструктора
                if constructor_option:
                    position_change -= 2
                    revenue_change += 1500

                # Влияние клуба
                if wb_club_discount:
                    position_change -= 1.5
                    revenue_change += 1200

                # Расчет новых значений
                current_pos = current_data.get("position_real")
                if current_pos is None:
                    current_pos = int(base_pos) if base_pos else 50

                # Округляем изменения позиции для лучшего отображения
                position_change = round(position_change)
                new_pos_sim = max(1, current_pos + position_change)
                new_revenue_sim = max(0, base_proc + revenue_change)

                st.metric(
                    "Новая позиция",
                    f"{int(new_pos_sim)}",
                    (
                        f"{position_change:+.0f} мест"
                        if abs(position_change) >= 0.5
                        else "Без изменений"
                    ),
                    delta_color="inverse" if position_change < 0 else "normal",
                )
                st.metric(
                    "Новая выручка",
                    f"{new_revenue_sim:,.0f}₽",
                    (
                        f"{revenue_change:+,.0f}₽"
                        if abs(revenue_change) >= 500
                        else "Без изменений"
                    ),
                )

                # Показываем детальную разбивку изменений
                if abs(position_change) >= 0.5 or abs(revenue_change) >= 500:
                    st.markdown("**📊 Детализация влияния:**")
                    
                    # Рассчитываем вклад каждого фактора для отображения
                    contributions = []
                    
                    base_promo = factors_data.get("Цена и скидки", 20)
                    if abs(promo_participation - base_promo) > 1:
                        promo_effect = -((promo_participation - base_promo) / 10.0) * 5
                        contributions.append(f"• Акции: {promo_effect:+.1f} поз.")
                    
                    base_sales = factors_data.get("Продажи и оборот", 45)
                    if abs(sales_volume - base_sales) > 1:
                        sales_effect = -((sales_volume - base_sales) / 15.0) * 8
                        contributions.append(f"• Продажи: {sales_effect:+.1f} поз.")
                    
                    base_rating = factors_data.get("Рейтинги и отзывы", 75)
                    if abs(product_rating - base_rating) > 1:
                        rating_effect = -((product_rating - base_rating) / 15.0) * 6
                        contributions.append(f"• Рейтинг: {rating_effect:+.1f} поз.")
                    
                    base_delivery = factors_data.get("Доставка", 60)
                    if abs(delivery_quality - base_delivery) > 1:
                        delivery_effect = -((delivery_quality - base_delivery) / 12.0) * 7
                        contributions.append(f"• Доставка: {delivery_effect:+.1f} поз.")
                    
                    current_price = current_data.get("price", 1200)
                    if abs(price - current_price) / current_price > 0.05:
                        price_effect = ((price - current_price) / current_price) * 15
                        contributions.append(f"• Цена: {price_effect:+.1f} поз.")
                    
                    if contributions:
                        for contrib in contributions[:4]:  # Показываем топ-4
                            st.markdown(contrib)
                
                # Показываем общий эффект
                if abs(position_change) >= 0.5 or abs(revenue_change) >= 500:
                    if position_change < -2:
                        st.success(f"🚀 Отличное улучшение: {abs(position_change):.0f} позиций вверх!")
                    elif position_change < -0.5:
                        st.success(f"📈 Улучшение позиции на {abs(position_change):.0f} мест")
                    elif position_change > 2:
                        st.warning(f"📉 Ухудшение позиции на {position_change:.0f} мест")
                    
                    if revenue_change > 5000:
                        st.success(f"💰 Рост выручки: +{revenue_change:,.0f}₽")
                    elif revenue_change < -1000:
                        st.warning(f"💸 Снижение выручки: {revenue_change:,.0f}₽")
                else:
                    st.info("🔄 Измените параметры больше для заметного эффекта")

            except Exception as e:
                st.error(f"Ошибка расчета: {e}")
                st.info("🔄 Попробуйте изменить параметры")
