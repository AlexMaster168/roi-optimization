import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from core.config import DEFAULT_ORG_TYPES

st.set_page_config(page_title="Порівняння", page_icon="📈", layout="wide")

st.header("📈 Порівняння організацій")

org_type = st.session_state.get("selected_org", DEFAULT_ORG_TYPES[0])
use_ml = st.session_state.get("use_ml", False)


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

available_orgs = DEFAULT_ORG_TYPES + st.session_state.get("dynamic_orgs", [])

if st.button("📊 Порівняти всі організації", type="primary"):
    with st.spinner("Порівняння організацій..."):
        all_results = []
        for org in available_orgs:
            try:
                prof = dg.get_profile(org)
                if use_ml and org == org_type and "predicted_probs" in st.session_state:
                    prbs = st.session_state["predicted_probs"]
                elif use_ml:
                    h_df = dg.generate_history(org, years=5)
                    from core.ml_forecaster import ThreatForecaster
                    fc = ThreatForecaster(model_type="auto")
                    fc.train(h_df, len(prof["threats"]))
                    p_sims = fc.predict_monte_carlo(h_df["year"].max() + 1, 1000)
                    prbs = np.array([np.mean(s) for s in p_sims])
                elif org == org_type and "pentest_adjusted_probs" in st.session_state:
                    prbs = st.session_state["pentest_adjusted_probs"]
                else:
                    prbs = np.array(prof["base_probs"])

                lss = np.array(prof["losses"])
                alpha = prof.get("alpha", 0.0)
                init_risk = optimizer.calculate_risk(prbs, lss)
                res = optimizer.optimize_brute_force(
                    prbs, lss, prof["protection_costs"], prof["budget_limit"], alpha=alpha,
                )

                if res["best_solution"]:
                    all_results.append({
                        "Організація": org,
                        "Початковий ризик": init_risk,
                        "Витрати": res["best_solution"]["cost"],
                        "Зниження (%)": res["best_solution"]["reduction_percent"],
                        "ROI": res["best_solution"]["roi"],
                    })
            except Exception as e:
                st.warning(f"Помилка для {org}: {e}")

    if all_results:
        df_compare = pd.DataFrame(all_results)
        st.dataframe(
            df_compare.style.format({
                "Початковий ризик": "{:,.0f}",
                "Витрати": "{:,.0f}",
                "Зниження (%)": "{:.1f}",
                "ROI": "{:.2f}",
            }),
            width="stretch",
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_roi = px.bar(
                df_compare, x="Організація", y="ROI",
                title="Порівняння ROI по організаціях",
                color="ROI", color_continuous_scale="Viridis",
            )
            st.plotly_chart(fig_roi, width="stretch")
        with col2:
            fig_reduction = px.bar(
                df_compare, x="Організація", y="Зниження (%)",
                title="Зниження ризику по організаціях",
                color="Зниження (%)", color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_reduction, width="stretch")
    else:
        st.warning("Немає даних для порівняння")
