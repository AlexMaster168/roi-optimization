import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_generator import CyberDataGenerator
from ml_forecaster import ThreatForecaster
from optimizer import BudgetOptimizer

st.set_page_config(
    page_title="CyberBudget AI Optimizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_data_generator():
    return CyberDataGenerator()


@st.cache_resource
def get_optimizer():
    return BudgetOptimizer()


dg = get_data_generator()
optimizer = get_optimizer()

st.sidebar.header("⚙️ Налаштування системи")

org_type = st.sidebar.selectbox(
    "Тип організації",
    ["E-commerce", "Bank", "Industry", "Healthcare", "Telecom", "University"]
)

use_ml = st.sidebar.checkbox(
    "🤖 Використовувати ML-прогнозування",
    value=True
)

model_type = st.sidebar.selectbox(
    "Тип ML-моделі",
    ["auto", "linear", "ridge", "lasso", "elastic_net", "random_forest", "gradient_boosting", "svr", "decision_tree"],
    disabled=not use_ml
)

n_simulations = st.sidebar.slider(
    "Кількість імітацій (Монте-Карло)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    disabled=not use_ml
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
    value=5
)

optimization_method = st.sidebar.radio(
    "Метод оптимізації",
    ["Brute Force (перебір)", "Continuous (scipy)"]
)

st.sidebar.markdown("---")

st.title("🛡️ Система оптимізації бюджету кібербезпеки з ML-прогнозуванням")

profile = dg.get_profile(org_type)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Інформація",
    "🤖 ML-Прогноз",
    "🎯 Оптимізація",
    "📈 Порівняння",
    "📑 Звіт"
])

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

with tab2:
    st.header("🤖 Прогнозування загроз за допомогою ML та Монте-Карло")

    if use_ml:
        history_df = dg.generate_history(org_type, years=history_years)

        forecaster = ThreatForecaster(model_type=model_type)
        forecaster.train(history_df, len(profile['threats']))

        last_year = history_df['year'].max()

        predicted_sims = forecaster.predict_monte_carlo(last_year + 1, n_simulations)
        pred_means = [np.mean(sims) for sims in predicted_sims]
        pred_lows = [np.percentile(sims, 5) for sims in predicted_sims]
        pred_highs = [np.percentile(sims, 95) for sims in predicted_sims]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Метрики базових моделей")
            metrics_df = forecaster.get_model_metrics(history_df)
            st.dataframe(metrics_df.style.format({
                'mse': '{:.6f}',
                'r2': '{:.4f}',
                'mae': '{:.6f}'
            }), use_container_width=True)

        with col2:
            st.subheader(f"Прогноз на рік {last_year + 1} (Монте-Карло)")
            df_forecast = pd.DataFrame({
                "Загроза": profile['threats'],
                "Поточна йм.": profile['base_probs'],
                "Прогноз (Mean)": pred_means,
                "Min (5%)": pred_lows,
                "Max (95%)": pred_highs
            })
            st.dataframe(df_forecast.style.format({
                'Поточна йм.': '{:.3f}',
                'Прогноз (Mean)': '{:.3f}',
                'Min (5%)': '{:.3f}',
                'Max (95%)': '{:.3f}'
            }), use_container_width=True)

        st.subheader("📈 Візуалізація прогнозу з довірчими інтервалами (90%)")
        fig_forecast = forecaster.plot_forecast(history_df, forecast_years, org_type, n_simulations)
        st.plotly_chart(fig_forecast, use_container_width=True)

        with st.expander("📊 Порівняння всіх регресійних моделей"):
            comp_df = forecaster.compare_all_models(history_df, len(profile['threats']))
            st.dataframe(comp_df.style.format({
                'MSE': '{:.6f}',
                'R2': '{:.4f}'
            }), use_container_width=True)

            fig_comp = px.bar(
                comp_df,
                x="Threat ID",
                y="R2",
                color="Model",
                barmode="group",
                title="Порівняння R2 моделей для кожної загрози"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.session_state['predicted_probs'] = np.array(pred_means)
        st.session_state['use_ml'] = True

    else:
        st.warning("⚠️ ML-прогнозування вимкнено.")
        st.session_state['predicted_probs'] = np.array(profile['base_probs'])
        st.session_state['use_ml'] = False

with tab3:
    st.header("🎯 Оптимізація розподілу бюджету")

    if st.session_state.get('use_ml', False) and 'predicted_probs' in st.session_state:
        probs_to_use = st.session_state['predicted_probs']
        st.info("🤖 Використовуються ML-прогнозовані ймовірності")
    else:
        probs_to_use = np.array(profile['base_probs'])
        st.warning("⚠️ Використовуються базові ймовірності")

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

with tab5:
    st.header("📑 Звіт для дисертації")

    if st.button("📥 Експортувати звіт"):
        st.success("Звіт готовий до експорту!")

st.markdown("---")
