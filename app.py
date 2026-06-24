import json
import logging

import streamlit as st

from core.config import DEFAULT_ORG_TYPES, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CyberBudget AI Оптимізатор",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dynamic_orgs" not in st.session_state:
    st.session_state["dynamic_orgs"] = []
if "selected_org" not in st.session_state:
    st.session_state["selected_org"] = DEFAULT_ORG_TYPES[0]

st.sidebar.header("⚙️ Налаштування системи")

st.sidebar.markdown("**Організація:** %s" % st.session_state["selected_org"])
st.sidebar.markdown("---")

org_type = st.sidebar.selectbox(
    "Тип організації",
    DEFAULT_ORG_TYPES + st.session_state["dynamic_orgs"],
    index=(DEFAULT_ORG_TYPES + st.session_state["dynamic_orgs"]).index(st.session_state["selected_org"])
    if st.session_state["selected_org"] in DEFAULT_ORG_TYPES + st.session_state["dynamic_orgs"]
    else 0,
)
st.session_state["selected_org"] = org_type

st.sidebar.markdown("---")
st.sidebar.caption("CyberBudget AI Оптимізатор v2.0")
st.sidebar.caption("ML-оптимізація бюджету кібербезпеки")

st.title("🛡️ CyberBudget AI Оптимізатор")
st.markdown("**Система оптимізації бюджету кібербезпеки з ML-прогнозуванням та пентестингом**")

col1, col2 = st.columns(2)
with col1:
    st.info("**📊 Дашборд** — інформація про профіль організації та загрози")
    st.info("**🤖 ML-Прогноз** — прогнозування загроз за допомогою машинного навчання")
with col2:
    st.info("**🎯 Оптимізація** — розподіл бюджету найоптимальнішим чином")
    st.info("**📈 Порівняння** — порівняння організацій між собою")
    st.info("**🔓 Пентест** — сканування та аналіз вразливостей")

st.markdown("---")
st.markdown("### Скористайтеся бічною панеллю для вибору організації, а потім перейдіть на будь-яку сторінку.")
