import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Локальные импорты должны быть в блоке try-except для совместимости со Streamlit Cloud
try:
    from src.wildbox_client import get_brand_details, get_product_details, get_warehouse_positions
    from src.wb_ranking_model import WBRankingModel
except ImportError:
    st.error("Не удалось импортировать локальные модули. Убедитесь, что файлы `wildbox_client.py` и `wb_ranking_model.py` находятся в папке `src/`.")
    st.stop()


def prepare_model_input(product_context, product_details, brand_details, city, search_query):
    """Собирает все данные в один словарь, готовый для модели."""
    subject_info = product_details.get("subject", {})
    data_dict = {
        "product_id": product_context.get("id"),
        "brand_id": product_details.get("brand", {}).get("id"),
        "category": subject_info.get("name") if isinstance(subject_info, dict) else None,
        "search_query": search_query,
        "city_of_search": city,
        "position": product_context.get("expected_position"),
        "proceeds": product_details.get("proceeds"),
        "delivery_efficiency_wh_avg_pos": float(product_context.get("wh_avg_position", 0) or 0),
        "in_promo": bool(product_details.get("promos")),
        "price": product_details.get("price"),
        "discount": product_details.get("discount"),
        "product_rating": product_details.get("rating"),
        "seller_rating": brand_details.get("rating"),
        "reviews_count": brand_details.get("reviews"),
        "quantity": product_details.get("quantity"),
        "lost_proceeds": product_details.get("lost_proceeds"),
        "images_count": len(product_details.get("images", [])),
        "product_name": product_details.get("name", "")
    }
    temp_df = pd.DataFrame([data_dict]).fillna(0)
    temp_df["is_discounted"] = (temp_df["discount"] > 0).astype(int)
    temp_df["price_per_rating"] = ((temp_df["price"] / temp_df["product_rating"].replace(0, 1))).fillna(0)
    return temp_df.iloc[0].to_dict()


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


def calculate_ranking_factors(current_data):
    """
    Рассчитывает оценку ключевых факторов ранжирования для товара (от 0 до 100).
    ВЕРСИЯ С ИСПРАВЛЕНИЕМ: Гарантирует, что все значения находятся в диапазоне [0, 100].
    """
    factors = {}
    
    # Цена и Скидки
    discount_score = 50 if current_data.get("is_discounted") else 0
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
    factors["Доставка"] = delivery_score
    
    # Остатки на складе (шкала до 500 шт)
    factors["Остатки на складе"] = min(100, (current_data.get("quantity", 0) / 500.0) * 100)

    # === ГЛАВНОЕ ИСПРАВЛЕНИЕ ===
    # Финальная проверка, чтобы гарантировать, что ни одно значение не выходит за пределы [0, 100]
    for key in factors:
        factors[key] = max(0, min(100, factors[key]))

    return factors


def generate_optimization_recommendations(current_data):
    """Генерирует умные рекомендации, всегда находя точки роста."""
    recommendations = []
    
    if current_data.get("discount", 0) < 15 and current_data.get("price", 0) > 0:
        recommendations.append({"factor": "Усилить акцию", "improvement": "Увеличить скидку до 20-25%", "position_change": "Значительный буст", "revenue_change": "Привлечение покупателей", "cost": f"~{int(current_data['price'] * 0.1):,}₽ доп. с единицы"})

    if calculate_constructor_score(current_data) < 90:
        recommendations.append({"factor": "Улучшить карточку", "improvement": "Добавить фото/видео, расширить описание", "position_change": "Рост CTR", "revenue_change": "Больше доверия и продаж", "cost": "от 5,000₽ (контент)"})

    if current_data.get("product_rating", 0) < 4.8:
        recommendations.append({"factor": "Поднять рейтинг", "improvement": f"С {current_data.get('product_rating', 0):.1f} до 4.8+ через работу с отзывами", "position_change": "Выше в поиске", "revenue_change": "Повышение лояльности", "cost": "от 10,000₽ (сервисы/качество)"})

    if current_data.get("proceeds", 0) < 50000:
        recommendations.append({"factor": "Увеличить продажи", "improvement": "Запустить внутреннюю рекламу (поиск/каталог)", "position_change": "Резкий рост видимости", "revenue_change": "Наращивание оборота", "cost": "от 15,000₽ (бюджет)"})
    
    if current_data.get("delivery_efficiency_wh_avg_pos", 100) > 48:
         recommendations.append({"factor": "Ускорить доставку", "improvement": "Распределить товар по региональным складам", "position_change": "Приоритет в выдаче", "revenue_change": "Рост заказов из регионов", "cost": "Логистические расходы"})

    if not recommendations:
        recommendations.append({"factor": "Анализ конкурентов", "improvement": "Найти слабые места у топ-5 и превзойти их", "position_change": "Стратегическое преимущество", "revenue_change": "Отстройка от рынка", "cost": "Время на анализ"})

    return recommendations[:4]


def get_full_report(city, search_query, product_id):
    """Основная функция для сбора данных, предсказания модели и возврата результатов."""
    try:
        product_details = get_product_details(product_id)
        if not product_details or "id" not in product_details:
            return None, None, None, f"Товар с артикулом {product_id} не найден."
        
        brand_id = product_details.get("brand", {}).get("id")
        brand_details = get_brand_details(brand_id) if brand_id else {}
        
        positions_data = get_warehouse_positions(product_id, search_query)
        city_pos_data = next((p for p in positions_data if p.get("city") == city), None)
        current_position = city_pos_data.get("position") if city_pos_data else None

        product_context = {"id": product_id, "expected_position": current_position, "wh_avg_position": product_details.get("wh_avg_position"), "brand": {"id": brand_id}}
        
        model_input_data = prepare_model_input(product_context, product_details, brand_details, city, search_query)
        if model_input_data.get("position") is None: model_input_data["position"] = 9999

        model_pos = st.session_state.model_pos
        model_proc = st.session_state.model_proc
        features = st.session_state.features
        
        product_df = pd.DataFrame([model_input_data])
        for feature in features:
            if feature not in product_df.columns: product_df[feature] = 0
        
        for col in ["city_of_search", "search_query", "category"]:
            if col in product_df.columns: product_df[col] = product_df[col].astype("category")

        base_position = np.expm1(model_pos.predict(product_df[features])[0])
        base_proceeds = model_proc.predict(product_df[features])[0]
        
        model_input_data["position_real"] = current_position
        return model_input_data, base_position, base_proceeds, None
    except Exception as e:
        return None, None, None, f"Ошибка при обработке данных: {e}"


# --- UI-часть приложения Streamlit ---
st.set_page_config(layout="wide", page_title="AI-Аналитик WB", page_icon="🤖")

st.title("🤖 AI-Аналитик для Wildberries")
st.markdown("**Анализ факторов ранжирования и симулятор для увеличения выручки**")

# Инициализация состояния приложения
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        product_id_input = st.text_input("Артикул товара (ID)", "32395071")
    with col2:
        search_query_input = st.text_input("Поисковый запрос", "футболка женская белая")
    with col3:
        city_input = st.selectbox("Город", ["Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань", "Краснодар"])

if st.button("🚀 Получить рекомендации", type="primary", use_container_width=True):
    if not product_id_input.isdigit():
        st.error("Артикул должен содержать только цифры.")
    else:
        with st.spinner("Загружаю модели и данные, анализирую..."):
            try:
                st.session_state.model_pos = joblib.load("models/position_model.pkl")
                st.session_state.model_proc = joblib.load("models/proceeds_model.pkl")
                st.session_state.features = joblib.load("models/feature_list.pkl")
                st.session_state.label_encoders = joblib.load("models/label_encoders.pkl")
            except FileNotFoundError:
                st.error("Файлы моделей не найдены. Убедитесь, что модели обучены и лежат в папке `models/`.")
                st.stop()
            
            current_data, base_pos, base_proc, error_message = get_full_report(city_input, search_query_input, int(product_id_input))

            if error_message:
                st.error(error_message)
                st.session_state.analysis_complete = False
            else:
                st.session_state.current_data = current_data
                st.session_state.base_pos = base_pos
                st.session_state.base_proc = base_proc
                st.session_state.analysis_complete = True
                st.success("Анализ завершен!")

if st.session_state.analysis_complete:
    current_data = st.session_state.current_data
    base_pos = st.session_state.base_pos
    base_proc = st.session_state.base_proc

    # --- ОСНОВНЫЕ МЕТРИКИ ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Текущая позиция", f"{int(current_data['position_real'])}" if current_data.get('position_real') is not None else "Не найден")
    col2.metric("📊 Факторов оценено", f"{len(calculate_ranking_factors(current_data))} из 6")
    col3.metric("💰 Прогноз выручки", f"{base_proc:,.0f}₽")
    col4.metric("💵 Цена", f"{current_data.get('price', 0):,.0f}₽", f"-{current_data.get('discount', 0)}%" if current_data.get('discount') else None)

    st.divider()

    # --- ФАКТОРЫ РАНЖИРОВАНИЯ ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📊 Влияние факторов на позицию")
        st.caption("Оценка AI-модели на основе ваших данных")
        try:
            with open("models/feature_importance.json", "r", encoding="utf-8") as f:
                tech_importances = json.load(f)
            
            factor_groups = {"Продажи и оборот": ["proceeds"], "Цена и скидки": ["price", "discount"], "Доставка": ["delivery"], "Рейтинги и отзывы": ["rating", "reviews"], "Качество карточки": ["images", "category", "query"], "Остатки на складе": ["quantity", "stock"], "Реклама": ["cpm", "ad_"]}
            agg_importance = {group: sum(imp for feat, imp in tech_importances.items() if any(k in feat for k in keys)) for group, keys in factor_groups.items()}
            
            total_sum = sum(agg_importance.values())
            if total_sum > 0:
                df = pd.DataFrame([{"Фактор": k, "Важность (%)": v / total_sum * 100} for k, v in agg_importance.items() if v > 0]).sort_values("Важность (%)")
                fig = px.bar(df, x='Важность (%)', y='Фактор', orientation="h", color='Важность (%)', color_continuous_scale=px.colors.sequential.Blues, text='Важность (%)')
                fig.update_layout(height=400, showlegend=False, xaxis_title=None, yaxis_title=None, margin=dict(l=0, r=0, t=0, b=0), xaxis_range=[0, df['Важность (%)'].max() * 1.1])
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Не удалось построить график важности факторов.")

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

    # --- РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ ---
    st.subheader("💡 Топ рекомендации по оптимизации")
    recommendations = generate_optimization_recommendations(current_data)
    rec_df = pd.DataFrame(recommendations).rename(columns={"factor": "ДЕЙСТВИЕ", "improvement": "ИЗМЕНЕНИЕ", "position_change": "ЭФФЕКТ НА ПОЗИЦИИ", "revenue_change": "ЭФФЕКТ НА ПРОДАЖИ", "cost": "СТОИМОСТЬ"})
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- СИМУЛЯТОР ИЗМЕНЕНИЙ ---
    st.subheader("🔧 Симулятор изменений параметров")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**Настройте параметры для симуляции:**")
        sim_price = st.slider("Цена товара, ₽", int(current_data.get("price", 1000) * 0.7), int(current_data.get("price", 1000) * 1.3), int(current_data.get("price", 1000)), key="sim_price")
        sim_discount = st.slider("Скидка, %", 0, 95, int(current_data.get("discount", 10)), key="sim_discount")
        sim_rating = st.slider("Рейтинг товара", 1.0, 5.0, float(current_data.get("product_rating", 4.5)), step=0.1, key="sim_rating")
        sim_images = st.slider("Кол-во изображений", 1, 15, int(current_data.get("images_count", 5)), key="sim_images")
        sim_proceeds_boost = st.slider("Усиление продаж (реклама), %", 0, 200, 0, key="sim_proceeds_boost")
    with col2:
        st.markdown("**Прогноз AI-модели:**")
        try:
            sim_data = current_data.copy()
            sim_data.update({'price': sim_price, 'discount': sim_discount, 'product_rating': sim_rating, 'images_count': sim_images, 'proceeds': sim_data.get('proceeds', 0) * (1 + sim_proceeds_boost / 100), 'in_promo': sim_discount > 0})
            
            model_utils = WBRankingModel()
            model_utils.label_encoders = st.session_state.label_encoders
            processed_sim_data = model_utils.prepare_data_from_unified_dataset(pd.DataFrame([sim_data]))
            X_sim = model_utils.prepare_features_for_model(processed_sim_data)
            
            for feature in st.session_state.features:
                if feature not in X_sim.columns: X_sim[feature] = 0
            X_sim = X_sim[st.session_state.features]
            
            new_pos_sim = int(np.expm1(st.session_state.model_pos.predict(X_sim)[0]))
            new_revenue_sim = st.session_state.model_proc.predict(X_sim)[0]
            
            st.metric("Прогноз позиции", f"~ {new_pos_sim}", f"{new_pos_sim - int(base_pos):+} мест", delta_color="inverse")
            st.metric("Прогноз выручки", f"{new_revenue_sim:,.0f} ₽", f"{new_revenue_sim - base_proc:+, .0f} ₽")
        except Exception as e:
            st.error(f"Ошибка симуляции: {e}")
