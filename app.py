import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_generator import CyberDataGenerator
from ml_forecaster import ThreatForecaster
from optimizer import BudgetOptimizer

# ==========================================
# КОНФІГУРАЦІЯ СТОРІНКИ
# ==========================================
st.set_page_config(
    page_title="CyberBudget AI Optimizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# ІНІЦІАЛІЗАЦІЯ
# ==========================================
@st.cache_resource
def get_data_generator():
    return CyberDataGenerator()


@st.cache_resource
def get_optimizer():
    return BudgetOptimizer()


dg = get_data_generator()
optimizer = get_optimizer()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Налаштування системи")

org_type = st.sidebar.selectbox(
    "Тип організації",
    ["E-commerce", "Bank", "Industry", "Healthcare", "Telecom", "University"],
    help="Оберіть профіль організації згідно зі статтею"
)

use_ml = st.sidebar.checkbox(
    "🤖 Використовувати ML-прогнозування",
    value=True,
    help="Інновація: регресійна модель для прогнозу ймовірностей"
)

model_type = st.sidebar.selectbox(
    "Тип ML-моделі",
    ["linear", "ridge", "random_forest"],
    disabled=not use_ml,
    help="Linear Regression, Ridge або Random Forest"
)

forecast_years = st.sidebar.slider(
    "Кількість років для прогнозу",
    min_value=1,
    max_value=5,
    value=3,
    disabled=not use_ml
)

history_years = st.sidebar.slider(
    "Кількість років історії",
    min_value=3,
    max_value=10,
    value=5,
    help="Дані для навчання ML-моделі"
)

optimization_method = st.sidebar.radio(
    "Метод оптимізації",
    ["Brute Force (перебір)", "Continuous (scipy)"],
    help="Перебір - як у статті, Continuous - інновація"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Наукова новизна:**
1. ML-прогнозування ймовірностей
2. Неперервна оптимізація
3. Динамічний аналіз ROI
""")

# ==========================================
# ГОЛОВНА ЧАСТИНА
# ==========================================
st.title("🛡️ Система оптимізації бюджету кібербезпеки з ML-прогнозуванням")
st.markdown("""
**Магістерська дисертація:** Розробка методики оптимізації розподілу бюджету на кібербезпеку 
на основі аналізу статистики загроз для організацій різних типів
""")

# Отримання профілю
profile = dg.get_profile(org_type)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Інформація",
    "🤖 ML-Прогноз",
    "🎯 Оптимізація",
    "📈 Порівняння",
    "📑 Звіт"
])

# ==========================================
# TAB 1: ІНФОРМАЦІЯ
# ==========================================
with tab1:
    st.header(f"Профіль: {org_type}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Бюджетний ліміт", f"{profile['budget_limit']:,} грн")
    col2.metric("Коефіцієнт α", f"{profile['alpha'] * 100:.0f}%")
    col3.metric("Кількість загроз", len(profile['threats']))

    st.subheader("📋 Параметри загроз")
    df_threats = pd.DataFrame({
        "Загроза": profile['threats'],
        "Базова ймовірність": profile['base_probs'],
        "Потенційні збитки (грн)": profile['losses']
    })
    df_threats['Очікуваний ризик'] = (df_threats['Базова ймовірність'] *
                                      df_threats['Потенційні збитки (грн)']).round(2)
    st.dataframe(df_threats.style.format({
        'Базова ймовірність': '{:.3f}',
        'Потенційні збитки (грн)': '{:,.0f}',
        'Очікуваний ризик': '{:,.0f}'
    }), use_container_width=True)

    # Графік базових ризиків
    fig = px.bar(
        df_threats,
        x="Загроза",
        y="Очікуваний ризик",
        color="Базова ймовірність",
        color_continuous_scale="Reds",
        title="Початковий розподіл ризиків по загрозах",
        labels={"Очікуваний ризик": "Ризик (грн)", "Базова ймовірність": "Ймовірність"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 2: ML-ПРОГНОЗ
# ==========================================
with tab2:
    st.header("🤖 Прогнозування загроз за допомогою ML")

    if use_ml:
        # Генерація історії
        history_df = dg.generate_history(org_type, years=history_years)

        # Навчання моделі
        forecaster = ThreatForecaster(model_type=model_type)
        forecaster.train(history_df, len(profile['threats']))

        # Прогноз
        last_year = history_df['year'].max()
        predicted_probs = forecaster.predict(last_year + 1)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Метрики моделей")
            metrics_df = forecaster.get_model_metrics(history_df)
            st.dataframe(metrics_df.style.format({
                'mse': '{:.6f}',
                'r2': '{:.4f}',
                'mae': '{:.6f}'
            }), use_container_width=True)

        with col2:
            st.subheader("Прогноз на наступний рік")
            df_forecast = pd.DataFrame({
                "Загроза": profile['threats'],
                "Поточна ймовірність": profile['base_probs'],
                "Прогнозована ймовірність": predicted_probs,
                "Зміна (%)": ((predicted_probs - profile['base_probs']) /
                              profile['base_probs'] * 100).round(2)
            })
            st.dataframe(df_forecast.style.format({
                'Поточна ймовірність': '{:.3f}',
                'Прогнозована ймовірність': '{:.3f}',
                'Зміна (%)': '{:+.1f}%'
            }), use_container_width=True)

        # Графік прогнозу
        st.subheader("📈 Візуалізація прогнозу")
        fig_forecast = forecaster.plot_forecast(history_df, forecast_years, org_type)
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Збереження прогнозованих ймовірностей для оптимізації
        st.session_state['predicted_probs'] = predicted_probs
        st.session_state['use_ml'] = True

    else:
        st.warning("⚠️ ML-прогнозування вимкнено. Використовуються базові ймовірності.")
        st.session_state['predicted_probs'] = np.array(profile['base_probs'])
        st.session_state['use_ml'] = False

# ==========================================
# TAB 3: ОПТИМІЗАЦІЯ
# ==========================================
with tab3:
    st.header("🎯 Оптимізація розподілу бюджету")

    # Вибір ймовірностей
    if st.session_state.get('use_ml', False) and 'predicted_probs' in st.session_state:
        probs_to_use = st.session_state['predicted_probs']
        st.info("🤖 Використовуються **ML-прогнозовані** ймовірності")
    else:
        probs_to_use = np.array(profile['base_probs'])
        st.warning("⚠️ Використовуються **базові** ймовірності")

    losses = np.array(profile['losses'])
    initial_risk = optimizer.calculate_risk(probs_to_use, losses)

    col1, col2 = st.columns(2)
    col1.metric("Початковий ризик", f"{initial_risk:,.0f} грн")
    col2.metric("Максимальний бюджет", f"{profile['budget_limit']:,} грн")

    if st.button("🚀 Розрахувати оптимальний розподіл", type="primary"):
        with st.spinner("Триває оптимізація..."):
            if optimization_method == "Brute Force (перебір)":
                result = optimizer.optimize_brute_force(
                    probs_to_use,
                    losses,
                    profile['protection_costs'],
                    profile['budget_limit']
                )
                best = result['best_solution']

                if best:
                    st.success(f"✅ Знайдено {result['total_solutions']} допустимих рішень")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Витрати", f"{best['cost']:,} грн")
                    col2.metric("Зниження ризику", f"{best['reduction_percent']:.1f}%")
                    col3.metric("Залишковий ризик", f"{best['new_risk']:,.0f} грн")
                    col4.metric("ROI", f"{best['roi']:.2f}:1")

                    # Таблиця розподілу
                    st.subheader("📋 Рекомендований розподіл бюджету")
                    df_distribution = pd.DataFrame({
                        "Загроза": profile['threats'],
                        "Рівень захисту": best['combination'],
                        "Витрати (грн)": [
                            profile['protection_costs'][i + 1][best['combination'][i]]
                            if best['combination'][i] != '0%' else 0
                            for i in range(len(profile['threats']))
                        ],
                        "Ймовірність до": probs_to_use,
                        "Ймовірність після": [
                            probs_to_use[i] * (1 - optimizer.reductions[best['combination'][i]])
                            for i in range(len(profile['threats']))
                        ]
                    })
                    st.dataframe(df_distribution.style.format({
                        'Витрати (грн)': '{:,.0f}',
                        'Ймовірність до': '{:.3f}',
                        'Ймовірність після': '{:.3f}'
                    }), use_container_width=True)

                    # Графік розподілу
                    fig_dist = px.bar(
                        df_distribution,
                        x="Загроза",
                        y="Витрати (грн)",
                        color="Рівень захисту",
                        title="Розподіл бюджету по загрозах",
                        color_discrete_sequence=px.colors.sequential.Blues
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

            else:
                result = optimizer.optimize_continuous(
                    probs_to_use,
                    losses,
                    profile['budget_limit']
                )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Витрати", f"{result['total_cost']:,.0f} грн")
                col2.metric("Зниження ризику", f"{result['reduction_percent']:.1f}%")
                col3.metric("Залишковий ризик", f"{result['residual_risk']:,.0f} грн")
                col4.metric("ROI", f"{result['roi']:.2f}:1")

                # Графік витрат
                fig_continuous = px.bar(
                    x=profile['threats'],
                    y=result['spending'],
                    labels={'x': 'Загроза', 'y': 'Витрати (грн)'},
                    title="Оптимальний розподіл бюджету (неперервна оптимізація)",
                    color=result['spending'],
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_continuous, use_container_width=True)

                st.session_state['continuous_result'] = result

# ==========================================
# TAB 4: ПОРІВНЯННЯ
# ==========================================
with tab4:
    st.header("⚖️ Порівняння методів")

    if st.button("📊 Порівняти всі організації"):
        all_results = []

        for org in dg.org_profiles.keys():
            profile = dg.get_profile(org)
            probs = np.array(profile['base_probs'])
            losses = np.array(profile['losses'])
            initial_risk = optimizer.calculate_risk(probs, losses)

            result = optimizer.optimize_brute_force(
                probs,
                losses,
                profile['protection_costs'],
                profile['budget_limit']
            )

            if result['best_solution']:
                all_results.append({
                    "Організація": org,
                    "Початковий ризик": initial_risk,
                    "Витрати": result['best_solution']['cost'],
                    "Зниження (%)": result['best_solution']['reduction_percent'],
                    "ROI": result['best_solution']['roi']
                })

        df_compare = pd.DataFrame(all_results)
        st.dataframe(df_compare.style.format({
            'Початковий ризик': '{:,.0f}',
            'Витрати': '{:,.0f}',
            'Зниження (%)': '{:.1f}',
            'ROI': '{:.2f}'
        }), use_container_width=True)

        # Графіки порівняння
        col1, col2 = st.columns(2)

        with col1:
            fig_roi = px.bar(
                df_compare,
                x="Організація",
                y="ROI",
                title="Порівняння ROI по організаціях",
                color="ROI",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_roi, use_container_width=True)

        with col2:
            fig_reduction = px.bar(
                df_compare,
                x="Організація",
                y="Зниження (%)",
                title="Зниження ризику по організаціях",
                color="Зниження (%)",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_reduction, use_container_width=True)

# ==========================================
# TAB 5: ЗВІТ
# ==========================================
with tab5:
    st.header("📑 Звіт для дисертації")

    st.markdown("""
    ### Наукова новизна розробленої системи:

    1. **Інтеграція ML-моделей** для динамічного прогнозування ймовірностей загроз
       - Лінійна регресія, Ridge, Random Forest
       - Автоматична оцінка якості моделей (R², MSE, MAE)

    2. **Два методи оптимізації**:
       - Перебір (класичний підхід зі статті)
       - Неперервна оптимізація (scipy.optimize) - інновація

    3. **Веб-інтерфейс** для практичного впровадження
       - Streamlit для швидкого розгортання
       - Інтерактивні графіки Plotly

    4. **Порівняльний аналіз** для 6 типів організацій
       - E-commerce, Bank, Industry, Healthcare, Telecom, University
    """)

    if st.button("📥 Експортувати звіт"):
        st.success("Звіт готовий до експорту! (функціонал можна додати)")

    st.markdown("""
    ---
    **Використані формули зі статті:**
    - Формула (1): Ймовірність загрози
    - Формула (2-3): Розрахунок ризику
    - Формула (4-5): Ефективність захисту
    - Формула (6): Допустимі витрати
    - Формула (7): ROI
    """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")