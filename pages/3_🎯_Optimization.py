import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from core.config import DEFAULT_ORG_TYPES

st.set_page_config(page_title="Оптимізація", page_icon="🎯", layout="wide")

st.header("🎯 Оптимізація розподілу бюджету")

org_type = st.session_state.get("selected_org", DEFAULT_ORG_TYPES[0])


@st.cache_resource
def get_data_generator():
    from core.data_generator import CyberDataGenerator
    return CyberDataGenerator()


@st.cache_resource
def get_optimizer():
    from core.optimizer import BudgetOptimizer
    return BudgetOptimizer()


dg = get_data_generator()
optimizer = get_optimizer()
profile = dg.get_profile(org_type)

if st.session_state.get("use_ml") and "predicted_probs" in st.session_state:
    probs_to_use = st.session_state["predicted_probs"]
    st.info("🤖 Використовуються ML-прогнозовані ймовірності")
elif "pentest_adjusted_probs" in st.session_state:
    probs_to_use = st.session_state["pentest_adjusted_probs"]
    st.info("🔓 Використовуються пентест-скориговані ймовірності")
else:
    probs_to_use = np.array(profile["base_probs"])
    st.warning("⚠️ Використовуються базові ймовірності")

losses = np.array(profile["losses"])
alpha = profile.get("alpha", 0.0)
initial_risk = optimizer.calculate_risk(probs_to_use, losses)

col1, col2, col3 = st.columns(3)
col1.metric("Початковий ризик", f"{initial_risk:,.0f} грн")
col2.metric("Максимальний бюджет", f"{profile['budget_limit']:,} грн")
col3.metric("Alpha (ризико-аверсія)", f"{alpha:.2f}")

optimization_method = st.radio(
    "Метод оптимізації",
    ["Brute Force (перебір)", "Continuous (scipy)"],
    horizontal=True,
)

if st.button("🚀 Розрахувати оптимальний розподіл", type="primary"):
    with st.spinner("Триває оптимізація..."):
        if optimization_method == "Brute Force (перебір)":
            result = optimizer.optimize_brute_force(
                probs_to_use, losses, profile["protection_costs"],
                profile["budget_limit"], alpha=alpha,
            )
            best = result["best_solution"]

            if best:
                st.success(f"✅ Знайдено {result['total_solutions']} допустимих рішень")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Витрати", f"{best['cost']:,} грн")
                col2.metric("Зниження ризику", f"{best['reduction_percent']:.1f}%")
                col3.metric("Залишковий ризик", f"{best['new_risk']:,.0f} грн")
                col4.metric("ROI", f"{best['roi']:.2f}:1")

                df_distribution = pd.DataFrame({
                    "Загроза": profile["threats"],
                    "Рівень захисту": best["combination"],
                    "Витрати (грн)": [
                        profile["protection_costs"][i + 1][best["combination"][i]]
                        if best["combination"][i] != "0%" else 0
                        for i in range(len(profile["threats"]))
                    ],
                    "Ймовірність до": probs_to_use,
                    "Ймовірність після": [
                        probs_to_use[i] * (1 - optimizer.reductions[best["combination"][i]])
                        for i in range(len(profile["threats"]))
                    ],
                })
                st.dataframe(
                    df_distribution.style.format({
                        "Витрати (грн)": "{:,.0f}",
                        "Ймовірність до": "{:.3f}",
                        "Ймовірність після": "{:.3f}",
                    }),
                    width="stretch",
                )

                fig_dist = px.bar(
                    df_distribution, x="Загроза", y="Витрати (грн)", color="Рівень захисту",
                    title="Розподіл бюджету по загрозах",
                    color_discrete_map={"0%": "#E0E0E0", "10%": "#A5D6A7", "30%": "#FFB74D", "80%": "#E57373"},
                )
                st.plotly_chart(fig_dist, width="stretch")

                st.session_state["last_optimization_result"] = {
                    "type": "brute",
                    "cost": best["cost"],
                    "new_risk": best["new_risk"],
                    "reduction_percent": best["reduction_percent"],
                    "reduction": best["reduction"],
                    "roi": best["roi"],
                    "combination": best["combination"],
                    "dist_df": df_distribution,
                }
            else:
                st.warning("⚠️ Жодне допустиме рішення не знайдено")
        else:
            result = optimizer.optimize_continuous(
                probs_to_use, losses, profile["budget_limit"], alpha=alpha,
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Витрати", f"{result['total_cost']:,.0f} грн")
            col2.metric("Зниження ризику", f"{result['reduction_percent']:.1f}%")
            col3.metric("Залишковий ризик", f"{result['residual_risk']:,.0f} грн")
            col4.metric("ROI", f"{result['roi']:.2f}:1")

            fig_continuous = px.bar(
                x=profile["threats"], y=result["spending"],
                labels={"x": "Загроза", "y": "Витрати (грн)"},
                title="Оптимальний розподіл бюджету",
                color=result["spending"], color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_continuous, width="stretch")

            st.session_state["last_optimization_result"] = {
                "type": "continuous",
                "cost": result["total_cost"],
                "new_risk": result["residual_risk"],
                "reduction_percent": result["reduction_percent"],
                "reduction": result["reduction"],
                "roi": result["roi"],
                "spending": result["spending"],
            }

with st.expander("📊 Аналіз чутливості (Sensitivity Analysis)"):
    if st.button("Провести аналіз чутливості"):
        with st.spinner("Аналіз чутливості..."):
            sensitivity = optimizer.sensitivity_analysis(
                probs_to_use, losses, profile["protection_costs"],
                profile["budget_limit"], alpha=alpha,
            )
            if sensitivity:
                df_sens = pd.DataFrame(sensitivity)
                fig_sens = px.line(
                    df_sens, x="budget", y="reduction",
                    labels={"budget": "Бюджет (грн)", "reduction": "Зниження ризику"},
                    title="Залежність зниження ризику від бюджету",
                    markers=True,
                )
                st.plotly_chart(fig_sens, width="stretch")

                fig_roi = px.line(
                    df_sens, x="budget", y="roi",
                    labels={"budget": "Бюджет (грн)", "roi": "ROI"},
                    title="Залежність ROI від бюджету",
                    markers=True,
                )
                st.plotly_chart(fig_roi, width="stretch")
            else:
                st.warning("Немає даних для аналізу чутливості")
