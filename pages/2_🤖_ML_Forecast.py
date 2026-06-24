import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from core.config import DEFAULT_ORG_TYPES

st.set_page_config(page_title="ML Прогноз", page_icon="🤖", layout="wide")

st.header("🤖 Прогнозування загроз за допомогою ML та Монте-Карло")

org_type = st.session_state.get("selected_org", DEFAULT_ORG_TYPES[0])

col1, col2 = st.columns(2)
with col1:
    use_ml = st.checkbox("Використовувати ML-прогнозування", value=True)
with col2:
    model_type = st.selectbox(
        "Тип ML-моделі",
        ["auto", "linear", "ridge", "lasso", "elastic_net", "random_forest", "gradient_boosting", "svr", "decision_tree"],
        disabled=not use_ml,
    )

col1, col2, col3 = st.columns(3)
with col1:
    n_simulations = st.slider("Кількість імітацій (Монте-Карло)", 100, 10000, 1000, 100, disabled=not use_ml)
with col2:
    forecast_years = st.slider("Кількість років для прогнозу", 1, 5, 3, disabled=not use_ml)
with col3:
    history_years = st.slider("Кількість років історії", 3, 10, 5)

if use_ml:
    with st.spinner("Триває навчання ML-моделей..."):
        @st.cache_resource
        def get_data_generator():
            from core.data_generator import CyberDataGenerator
            return CyberDataGenerator()

        dg = get_data_generator()
        profile = dg.get_profile(org_type)
        history_df = dg.generate_history(org_type, years=history_years)

        from core.ml_forecaster import ThreatForecaster
        forecaster = ThreatForecaster(model_type=model_type)
        forecaster.train(history_df, len(profile["threats"]))
        last_year = history_df["year"].max()

        predicted_sims = forecaster.predict_monte_carlo(last_year + 1, n_simulations)
        pred_means = [np.mean(sims) for sims in predicted_sims]
        pred_lows = [np.percentile(sims, 5) for sims in predicted_sims]
        pred_highs = [np.percentile(sims, 95) for sims in predicted_sims]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Метрики базових моделей")
        metrics_df = forecaster.get_model_metrics(history_df)
        st.dataframe(
            metrics_df.style.format({"mse": "{:.6f}", "r2": "{:.4f}", "mae": "{:.6f}", "cv_r2": "{:.4f}"}),
            width="stretch",
        )

    with col2:
        st.subheader(f"🔮 Прогноз на рік {last_year + 1} (Монте-Карло)")
        df_forecast = pd.DataFrame({
            "Загроза": profile["threats"],
            "Поточна йм.": profile["base_probs"],
            "Прогноз (Mean)": pred_means,
            "Min (5%)": pred_lows,
            "Max (95%)": pred_highs,
        })
        st.dataframe(
            df_forecast.style.format({
                "Поточна йм.": "{:.3f}",
                "Прогноз (Mean)": "{:.3f}",
                "Min (5%)": "{:.3f}",
                "Max (95%)": "{:.3f}",
            }),
            width="stretch",
        )

    st.subheader("📈 Візуалізація прогнозу з довірчими інтервалами (90%)")
    fig_forecast = forecaster.plot_forecast(history_df, forecast_years, org_type, n_simulations)
    st.plotly_chart(fig_forecast, width="stretch")

    with st.expander("📊 Порівняння всіх регресійних моделей"):
        comp_df = forecaster.compare_all_models(history_df, len(profile["threats"]))
        st.dataframe(comp_df.style.format({"MSE": "{:.6f}", "R2": "{:.4f}"}), width="stretch")
        fig_comp = px.bar(
            comp_df, x="Threat ID", y="R2", color="Model", barmode="group",
            title="Порівняння R2 моделей для кожної загрози",
        )
        st.plotly_chart(fig_comp, width="stretch")

    st.session_state["predicted_probs"] = np.array(pred_means)
    st.session_state["use_ml"] = True
else:
    st.warning("⚠️ ML-прогнозування вимкнено.")
    @st.cache_resource
    def get_data_generator_fallback():
        from core.data_generator import CyberDataGenerator
        return CyberDataGenerator()

    dg_fb = get_data_generator_fallback()
    profile_fb = dg_fb.get_profile(org_type)
    st.session_state["predicted_probs"] = np.array(profile_fb["base_probs"])
    st.session_state["use_ml"] = False
