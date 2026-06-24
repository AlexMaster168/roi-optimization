import json

import streamlit as st
import pandas as pd
import plotly.express as px

from core.config import DEFAULT_ORG_TYPES

st.set_page_config(page_title="Дашборд", page_icon="📊", layout="wide")


@st.cache_resource
def get_data_generator():
    from core.data_generator import CyberDataGenerator
    return CyberDataGenerator()


dg = get_data_generator()

st.header("📊 Профіль організації")

org_type = st.session_state.get("selected_org", DEFAULT_ORG_TYPES[0])

uploaded_profile = st.file_uploader("Завантаження профілю організації (JSON)", type=["json"])
if uploaded_profile is not None:
    try:
        custom_profile = json.loads(uploaded_profile.getvalue().decode("utf-8"))
        org_name = custom_profile.get("org_name", "Custom Organization")
        profile_data = {
            "threats": custom_profile.get("threats", []),
            "base_probs": custom_profile.get("base_probs", []),
            "losses": custom_profile.get("losses", []),
            "protection_costs": custom_profile.get("protection_costs", {}),
            "budget_limit": custom_profile.get("budget_limit", 0),
            "alpha": custom_profile.get("alpha", 0.1),
        }
        dg.load_custom_profile(org_name, profile_data)
        if org_name not in st.session_state.get("dynamic_orgs", []):
            st.session_state.setdefault("dynamic_orgs", []).append(org_name)
        st.success(f"Профіль {org_name} завантажено!")
        org_type = org_name
    except Exception as e:
        st.error(f"Помилка завантаження: {e}")

if org_type not in dg.org_profiles:
    org_type = DEFAULT_ORG_TYPES[0]

profile = dg.get_profile(org_type)

col1, col2, col3 = st.columns(3)
col1.metric("Бюджетний ліміт", f"{profile['budget_limit']:,} грн")
col2.metric("Коефіцієнт α", f"{profile['alpha'] * 100:.0f}%")
col3.metric("Кількість загроз", len(profile["threats"]))

st.subheader("📋 Параметри загроз")
df_threats = pd.DataFrame({
    "Загроза": profile["threats"],
    "Базова ймовірність": profile["base_probs"],
    "Потенційні збитки (грн)": profile["losses"],
})
df_threats["Очікуваний ризик"] = (
    df_threats["Базова ймовірність"] * df_threats["Потенційні збитки (грн)"]
).round(2)

st.dataframe(
    df_threats.style.format({
        "Базова ймовірність": "{:.3f}",
        "Потенційні збитки (грн)": "{:,.0f}",
        "Очікуваний ризик": "{:,.0f}",
    }),
    width="stretch",
)

fig = px.bar(
    df_threats,
    x="Загроза",
    y="Очікуваний ризик",
    color="Базова ймовірність",
    color_continuous_scale="Reds",
    title="Початковий розподіл ризиків по загрозах",
)
st.plotly_chart(fig, width="stretch")

if org_type in st.session_state.get("dynamic_orgs", []):
    if st.button(f"🗑️ Видалити {org_type}"):
        dg.remove_custom_profile(org_type)
        st.session_state["dynamic_orgs"].remove(org_type)
        st.session_state["selected_org"] = DEFAULT_ORG_TYPES[0]
        st.rerun()
