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


def calculate_ranking_factors(current_data):
    """Рассчитывает факторы ранжирования на основе РЕАЛЬНЫХ данных товара."""
    factors = {}

    if current_data.get("is_discounted"):
        factors["Акции"] = current_data.get("discount", 0)

    if current_data.get("proceeds") is not None:
        factors["Продажи"] = min(100, current_data.get("proceeds", 0) / 1000)

    if current_data.get("seller_rating") is not None:
        factors["Рейтинг продавца"] = current_data.get("seller_rating", 0) * 20

    if current_data.get("product_rating") is not None:
        factors["Рейтинг товара"] = current_data.get("product_rating", 0) * 20

    # Конструктор на основе реальных данных карточки
    constructor_score = calculate_constructor_score(current_data)
    if constructor_score > 0:
        factors["Качество карточки"] = constructor_score

    if current_data.get("delivery_efficiency_wh_avg_pos") is not None:
        delivery_score = min(
            100,
            max(
                0,
                100
                - (current_data.get("delivery_efficiency_wh_avg_pos", 0) / 100 * 100),
            ),
        )
        if delivery_score > 0:
            factors["Доставка"] = delivery_score

    return factors


def calculate_constructor_score(product_data):
    """Рассчитывает качество использования конструктора на основе данных карточки."""
    score = 0

    # Проверяем заполненность описания
    description = product_data.get("description", "")
    if isinstance(description, str) and len(description) > 200:
        score += 25
    elif isinstance(description, str) and len(description) > 100:
        score += 15

    # Проверяем количество изображений
    images_count = product_data.get("images_count", 0)
    if images_count >= 8:
        score += 25
    elif images_count >= 5:
        score += 15
    elif images_count >= 3:
        score += 10

    # Проверяем качество названия
    product_name = str(product_data.get("product_name", ""))
    name_words = len(product_name.split())
    if name_words >= 5:  # Хорошо оптимизированное название
        score += 25
    elif name_words >= 3:
        score += 15

    # Проверяем наличие категории
    if product_data.get("category"):
        score += 25

    # Ограничиваем максимальный скор
    return min(100, score)


def generate_optimization_recommendations(current_data, base_pos, base_proc):
    """Генерирует рекомендации"""
    recommendations = []

    # Проверяем акции
    if not current_data.get("is_discounted", 0) and current_data.get("price"):
        current_price = current_data.get("price")
        discount_cost = current_price * 0.15  # 15% скидка от цены товара
        recommendations.append(
            {
                "factor": "Запустить акцию",
                "improvement": "Скидка 15%",
                "position_change": "Улучшение позиций",
                "revenue_change": "Рост конверсии",
                "cost": f"{int(discount_cost):,}₽ потеря маржи",
            }
        )

    # Проверяем изображения
    images_count = current_data.get("images_count", 0)
    if images_count < 5:
        recommendations.append(
            {
                "factor": "Улучшить карточку",
                "improvement": f"Добавить {5-images_count} фото",
                "position_change": "Лучше конверсия",
                "revenue_change": "Больше продаж",
                "cost": "8,000₽ фотосъемка",
            }
        )

    # Проверяем рейтинг
    rating = current_data.get("product_rating", 0)
    if rating > 0 and rating < 4.5:
        recommendations.append(
            {
                "factor": "Поднять рейтинг",
                "improvement": f"С {rating:.1f} до 4.5+",
                "position_change": "Выше в поиске",
                "revenue_change": "Больше доверия",
                "cost": "10,000₽ работа с качеством",
            }
        )

    # Проверяем рподажи
    if not current_data.get("proceeds"):
        recommendations.append(
            {
                "factor": "Запустить продвижение",
                "improvement": "Начать продажи",
                "position_change": "Появление в поиске",
                "revenue_change": "Первые продажи",
                "cost": "20,000₽ реклама + промо",
            }
        )

    return recommendations


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
            model_pos = joblib.load("position_model.pkl")
            model_proc = joblib.load("proceeds_model.pkl")
            features = joblib.load("feature_list.pkl")
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
                    value=f"{factors_count} из 7",
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

        # === ФАКТОРЫ РАНЖИРОВАНИЯ ===
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊 Влияние факторов на ранжирование")
            st.caption(
                "Теоретическая важность факторов в алгоритме WB (по исследованиям)"
            )

            # Теоретические данные о важности факторов (из публичных исследований)
            importance_data = {
                "Доставка": 35,
                "Продажи/Оборот": 25,
                "Конверсия/CTR": 15,
                "Рейтинги": 10,
                "Акции/Промо": 8,
                "SEO карточки": 4,
                "Прочее": 3,
            }

            factor_names = list(importance_data.keys())
            factor_values = list(importance_data.values())

            fig_importance = px.bar(
                x=factor_values,
                y=factor_names,
                orientation="h",
                title="",
                color=factor_values,
                color_continuous_scale="Blues",
            )
            fig_importance.update_layout(
                height=400,
                showlegend=False,
                coloraxis_showscale=False,
                xaxis_title="Влияние на ранжирование",
                yaxis_title="",
                margin=dict(l=0, r=0, t=0, b=0),
            )
            fig_importance.update_traces(texttemplate="%{x}", textposition="outside")
            st.plotly_chart(fig_importance, use_container_width=True)

        with col2:
            st.subheader("🎯 Факторы вашего товара")
            st.caption("На основе реальных данных из API")

            # Прогресс-бары для факторов
            factors_data = calculate_ranking_factors(current_data)

            if not factors_data:
                st.warning("⚠️ Недостаточно данных для анализа факторов")
                st.info(
                    "Возможные причины: товар новый, данные не полные, ограничения API"
                )
            else:
                for factor, value in factors_data.items():
                    # Цветовая схема
                    if value >= 70:
                        color = "🟢"
                    elif value >= 40:
                        color = "🟡"
                    else:
                        color = "🔴"

                    st.markdown(f"{color} **{factor}**")
                    st.progress(value / 100)
                    st.markdown(
                        f"<div style='text-align: right; margin-top: -20px; margin-bottom: 10px;'><small>{value:.0f}%</small></div>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # === РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ ===
        st.subheader("💡 Топ рекомендации по оптимизации")

        recommendations = generate_optimization_recommendations(
            current_data, base_pos, base_proc
        )

        # Таблица рекомендаций
        rec_df = pd.DataFrame(recommendations)
        if not rec_df.empty:
            rec_df = rec_df.rename(
                columns={
                    "factor": "ДЕЙСТВИЕ",
                    "improvement": "ИЗМЕНЕНИЕ",
                    "position_change": "ЭФФЕКТ НА ПОЗИЦИИ",
                    "revenue_change": "ЭФФЕКТ НА ПРОДАЖИ",
                    "cost": "СТОИМОСТЬ",
                }
            )

            st.dataframe(rec_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "🎉 Все основные факторы оптимизированы! Товар в хорошем состоянии."
            )

        st.divider()

        # === РЕЗУЛЬТАТЫ КОМПЛЕКСНОЙ ОПТИМИЗАЦИИ ===
        if recommendations:
            st.subheader("🎯 Потенциал улучшения")

            col1, col2 = st.columns(2)

            with col1:
                st.info("📈 **Возможные улучшения:**")
                for rec in recommendations[:3]:  # Показываем топ-3
                    st.write(f"• {rec['factor']}: {rec['improvement']}")

            with col2:
                st.info("💰 **Инвестиции в оптимизацию:**")
                total_cost = sum(
                    [
                        int(rec["cost"].replace("₽", "").replace(",", "").split()[0])
                        for rec in recommendations
                        if "₽" in rec["cost"]
                    ]
                )
                if total_cost > 0:
                    st.write(f"• Общие затраты: ~{total_cost:,}₽")
                    st.write(
                        f"• Средняя стоимость действия: {total_cost//len(recommendations):,}₽"
                    )
        else:
            st.success("🎉 Товар хорошо оптимизирован!")
            st.info("Основные факторы ранжирования находятся в хорошем состоянии.")

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
                max(20, int(factors_data.get("Акции", 20))),
                key="promo_slider",
            )
            sales_volume = st.slider(
                "Объем продаж:",
                0,
                100,
                max(45, int(factors_data.get("Продажи", 45))),
                key="sales_slider",
            )
            conversion = st.slider(
                "Конверсия:",
                0,
                100,
                max(60, int(factors_data.get("Конверсия", 60))),
                key="conversion_slider",
            )
            product_rating = st.slider(
                "Рейтинг товара:",
                0,
                100,
                max(75, int(factors_data.get("Рейтинг товара", 75))),
                key="rating_slider",
            )
            card_completion = st.slider(
                "Наполнение карточки:", 50, 100, 70, key="card_slider"
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
                # Влияние акций
                base_promo = factors_data.get("Акции", 20)
                if promo_participation > base_promo:
                    diff = (promo_participation - base_promo) / 20.0
                    position_change -= int(diff * 3)
                    revenue_change += int(diff * 4000)

                # Влияние рейтинга
                base_rating = factors_data.get("Рейтинг товара", 75)
                if product_rating > base_rating:
                    diff = (product_rating - base_rating) / 25.0
                    position_change -= int(diff * 2)
                    revenue_change += int(diff * 2000)

                # Влияние конверсии
                base_conversion = factors_data.get("Конверсия", 60)
                if conversion > base_conversion:
                    diff = (conversion - base_conversion) / 20.0
                    position_change -= int(diff * 2)
                    revenue_change += int(diff * 3000)

                # Влияние конструктора
                if constructor_option:
                    position_change -= 1
                    revenue_change += 1000

                # Влияние клуба
                if wb_club_discount:
                    position_change -= 1
                    revenue_change += 800

                # Расчет новых значений
                current_pos = current_data.get("position_real")
                if current_pos is None:
                    current_pos = int(base_pos) if base_pos else 50

                new_pos_sim = max(1, current_pos + position_change)
                new_revenue_sim = base_proc + revenue_change

                st.metric(
                    "Новая позиция",
                    f"~ {int(new_pos_sim)}",
                    (
                        f"{position_change} мест"
                        if position_change != 0
                        else "Без изменений"
                    ),
                    delta_color="inverse" if position_change < 0 else "normal",
                )
                st.metric(
                    "Новая выручка",
                    f"~ {new_revenue_sim:,.0f}₽",
                    (
                        f"+{revenue_change:,.0f}₽"
                        if revenue_change > 0
                        else "Без изменений"
                    ),
                )

                # Показываем общий эффект
                if position_change != 0 or revenue_change > 0:
                    st.success(
                        f"📈 Общий эффект: {abs(position_change)} позиций вверх, +{revenue_change:,.0f}₽ к выручке"
                    )
                else:
                    st.info("🔄 Измените параметры выше для просмотра эффекта")

            except Exception as e:
                st.error(f"Ошибка расчета: {e}")
                st.info("🔄 Попробуйте изменить параметры")
