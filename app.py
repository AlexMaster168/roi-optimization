import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from data_generator import CyberDataGenerator
from ml_forecaster import ThreatForecaster
from optimizer import BudgetOptimizer
from scanner_integration import OwaspReportParser

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


@st.cache_resource
def get_parser():
    return OwaspReportParser()


dg = get_data_generator()
optimizer = get_optimizer()
parser = get_parser()

if 'dynamic_orgs' not in st.session_state:
    st.session_state['dynamic_orgs'] = []

st.sidebar.header("⚙️ Налаштування системи")

with st.sidebar.expander("📂 Завантаження зовнішніх даних"):
    uploaded_profile = st.file_uploader("Профіль організації (JSON)", type=["json"], key="profile_upload")
    uploaded_scan = st.file_uploader("Звіт OWASP ZAP (JSON)", type=["json"], key="scan_upload")

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
            "alpha": custom_profile.get("alpha", 0.1)
        }
        dg.load_custom_profile(org_name, profile_data)
        if org_name not in st.session_state['dynamic_orgs']:
            st.session_state['dynamic_orgs'].append(org_name)
        st.sidebar.success(f"✅ Профіль {org_name} завантажено!")
    except Exception as e:
        st.sidebar.error(f"Помилка завантаження профілю: {e}")

if uploaded_scan is not None:
    try:
        file_content = uploaded_scan.getvalue().decode("utf-8")
        st.session_state['scan_df'] = parser.parse_zap_json(file_content)
        if "OWASP_Top_10_Web" not in st.session_state['dynamic_orgs']:
            st.session_state['dynamic_orgs'].append("OWASP_Top_10_Web")
        st.sidebar.success("✅ Звіт OWASP ZAP успішно враховано!")
    except Exception as e:
        st.sidebar.error(f"Помилка обробки файлу ZAP: {e}")

base_orgs = ["E-commerce", "Bank", "Industry", "Healthcare", "Telecom", "University"]
available_orgs = base_orgs + st.session_state['dynamic_orgs']

if 'selected_org' not in st.session_state or st.session_state['selected_org'] not in available_orgs:
    st.session_state['selected_org'] = available_orgs[0]

org_type = st.sidebar.selectbox(
    "Тип організації",
    available_orgs,
    index=available_orgs.index(st.session_state['selected_org'])
)
st.session_state['selected_org'] = org_type

if org_type in st.session_state['dynamic_orgs']:
    if st.sidebar.button(f"🗑️ Видалити {org_type}"):
        dg.remove_custom_profile(org_type)
        st.session_state['dynamic_orgs'].remove(org_type)
        if org_type == "OWASP_Top_10_Web" and 'scan_df' in st.session_state:
            del st.session_state['scan_df']
        st.session_state['selected_org'] = base_orgs[0]
        st.rerun()

use_ml = st.sidebar.checkbox("🤖 Використовувати ML-прогнозування", value=True)

model_type = st.sidebar.selectbox(
    "Тип ML-моделі",
    ["auto", "linear", "ridge", "lasso", "elastic_net", "random_forest", "gradient_boosting", "svr", "decision_tree"],
    disabled=not use_ml
)

n_simulations = st.sidebar.slider("Кількість імітацій (Монте-Карло)", 100, 10000, 1000, 100, disabled=not use_ml)
forecast_years = st.sidebar.slider("Кількість років для прогнозу", 1, 5, 3, disabled=not use_ml)
history_years = st.sidebar.slider("Кількість років історії", 3, 10, 5)

optimization_method = st.sidebar.radio(
    "Метод оптимізації",
    ["Brute Force (перебір)", "Continuous (scipy)"]
)

st.sidebar.markdown("---")
st.title("🛡️ Система оптимізації бюджету кібербезпеки з ML-прогнозуванням")

profile = dg.get_profile(org_type)
base_probs = profile['base_probs'].copy()

if org_type == "OWASP_Top_10_Web" and 'scan_df' in st.session_state:
    base_probs = parser.adjust_probabilities(base_probs, profile['threats'], st.session_state['scan_df'])

profile['base_probs'] = base_probs

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Інформація", "🤖 ML-Прогноз", "🎯 Оптимізація", "📈 Порівняння", "📑 Звіт"])

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
    df_threats['Очікуваний ризик'] = (df_threats['Базова ймовірність'] * df_threats['Потенційні збитки (грн)']).round(2)
    st.dataframe(df_threats.style.format({
        'Базова ймовірність': '{:.3f}',
        'Потенційні збитки (грн)': '{:,.0f}',
        'Очікуваний ризик': '{:,.0f}'
    }), use_container_width=True)

    fig = px.bar(
        df_threats, x="Загроза", y="Очікуваний ризик", color="Базова ймовірність",
        color_continuous_scale="Reds", title="Початковий розподіл ризиків по загрозах"
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
            st.dataframe(metrics_df.style.format({'mse': '{:.6f}', 'r2': '{:.4f}', 'mae': '{:.6f}'}),
                         use_container_width=True)

        with col2:
            st.subheader(f"Прогноз на рік {last_year + 1} (Монте-Карло)")
            df_forecast = pd.DataFrame({
                "Загроза": profile['threats'], "Поточна йм.": profile['base_probs'],
                "Прогноз (Mean)": pred_means, "Min (5%)": pred_lows, "Max (95%)": pred_highs
            })
            st.dataframe(df_forecast.style.format({
                'Поточна йм.': '{:.3f}', 'Прогноз (Mean)': '{:.3f}',
                'Min (5%)': '{:.3f}', 'Max (95%)': '{:.3f}'
            }), use_container_width=True)

        st.subheader("📈 Візуалізація прогнозу з довірчими інтервалами (90%)")
        fig_forecast = forecaster.plot_forecast(history_df, forecast_years, org_type, n_simulations)
        st.plotly_chart(fig_forecast, use_container_width=True)

        with st.expander("📊 Порівняння всіх регресійних моделей"):
            comp_df = forecaster.compare_all_models(history_df, len(profile['threats']))
            st.dataframe(comp_df.style.format({'MSE': '{:.6f}', 'R2': '{:.4f}'}), use_container_width=True)
            fig_comp = px.bar(comp_df, x="Threat ID", y="R2", color="Model", barmode="group",
                              title="Порівняння R2 моделей для кожної загрози")
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
                result = optimizer.optimize_brute_force(probs_to_use, losses, profile['protection_costs'],
                                                        profile['budget_limit'])
                best = result['best_solution']

                if best:
                    st.success(f"✅ Знайдено {result['total_solutions']} допустимих рішень")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Витрати", f"{best['cost']:,} грн")
                    col2.metric("Зниження ризику", f"{best['reduction_percent']:.1f}%")
                    col3.metric("Залишковий ризик", f"{best['new_risk']:,.0f} грн")
                    col4.metric("ROI", f"{best['roi']:.2f}:1")

                    df_distribution = pd.DataFrame({
                        "Загроза": profile['threats'], "Рівень захисту": best['combination'],
                        "Витрати (грн)": [
                            profile['protection_costs'][i + 1][best['combination'][i]] if best['combination'][
                                                                                              i] != '0%' else 0 for i in
                            range(len(profile['threats']))],
                        "Ймовірність до": probs_to_use,
                        "Ймовірність після": [probs_to_use[i] * (1 - optimizer.reductions[best['combination'][i]]) for i
                                              in range(len(profile['threats']))]
                    })
                    st.dataframe(df_distribution.style.format(
                        {'Витрати (грн)': '{:,.0f}', 'Ймовірність до': '{:.3f}', 'Ймовірність після': '{:.3f}'}),
                        use_container_width=True)

                    fig_dist = px.bar(
                        df_distribution, x="Загроза", y="Витрати (грн)", color="Рівень захисту",
                        title="Розподіл бюджету по загрозах",
                        color_discrete_map={'0%': '#E0E0E0', '10%': '#A5D6A7', '30%': '#FFB74D', '80%': '#E57373'}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    st.session_state['last_optimization_result'] = {
                        'type': 'brute', 'cost': best['cost'], 'new_risk': best['new_risk'],
                        'reduction_percent': best['reduction_percent'], 'reduction': best['reduction'],
                        'roi': best['roi'], 'combination': best['combination'], 'dist_df': df_distribution
                    }
            else:
                result = optimizer.optimize_continuous(probs_to_use, losses, profile['budget_limit'])
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Витрати", f"{result['total_cost']:,.0f} грн")
                col2.metric("Зниження ризику", f"{result['reduction_percent']:.1f}%")
                col3.metric("Залишковий ризик", f"{result['residual_risk']:,.0f} грн")
                col4.metric("ROI", f"{result['roi']:.2f}:1")

                fig_continuous = px.bar(
                    x=profile['threats'], y=result['spending'], labels={'x': 'Загроза', 'y': 'Витрати (грн)'},
                    title="Оптимальний розподіл бюджету", color=result['spending'], color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_continuous, use_container_width=True)

                st.session_state['last_optimization_result'] = {
                    'type': 'continuous', 'cost': result['total_cost'], 'new_risk': result['residual_risk'],
                    'reduction_percent': result['reduction_percent'], 'reduction': result['reduction'],
                    'roi': result['roi'], 'spending': result['spending']
                }

with tab4:
    st.header("⚖️ Порівняння методів")
    if st.button("📊 Порівняти всі організації"):
        all_results = []
        for org in available_orgs:
            prof = dg.get_profile(org)
            if use_ml:
                if org == org_type:
                    prbs = probs_to_use
                else:
                    h_df = dg.generate_history(org, years=history_years)
                    fc = ThreatForecaster(model_type=model_type)
                    fc.train(h_df, len(prof['threats']))
                    p_sims = fc.predict_monte_carlo(h_df['year'].max() + 1, n_simulations)
                    prbs = np.array([np.mean(s) for s in p_sims])
            else:
                prbs = np.array(prof['base_probs'])
                if org == "OWASP_Top_10_Web" and 'scan_df' in st.session_state:
                    prbs = np.array(parser.adjust_probabilities(prbs, prof['threats'], st.session_state['scan_df']))

            lss = np.array(prof['losses'])
            init_risk = optimizer.calculate_risk(prbs, lss)
            res = optimizer.optimize_brute_force(prbs, lss, prof['protection_costs'], prof['budget_limit'])

            if res['best_solution']:
                all_results.append({
                    "Організація": org, "Початковий ризик": init_risk,
                    "Витрати": res['best_solution']['cost'], "Зниження (%)": res['best_solution']['reduction_percent'],
                    "ROI": res['best_solution']['roi']
                })

        df_compare = pd.DataFrame(all_results)
        st.dataframe(df_compare.style.format(
            {'Початковий ризик': '{:,.0f}', 'Витрати': '{:,.0f}', 'Зниження (%)': '{:.1f}', 'ROI': '{:.2f}'}),
            use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_roi = px.bar(df_compare, x="Організація", y="ROI", title="Порівняння ROI по організаціях", color="ROI",
                             color_continuous_scale="Viridis")
            st.plotly_chart(fig_roi, use_container_width=True)
        with col2:
            fig_reduction = px.bar(df_compare, x="Організація", y="Зниження (%)",
                                   title="Зниження ризику по організаціях", color="Зниження (%)",
                                   color_continuous_scale="Reds")
            st.plotly_chart(fig_reduction, use_container_width=True)

with tab5:
    st.header("📑 Детальний звіт та висновки")
    if 'last_optimization_result' in st.session_state:
        res = st.session_state['last_optimization_result']
        report_md = f"""# Детальний звіт з оптимізації кібербезпеки

**Організація:** {org_type}
**Використання ML-прогнозування:** {'Так' if use_ml else 'Ні'}
**Метод оптимізації:** {optimization_method}

## 1. Початкові показники
* **Загальний ліміт бюджету:** {profile['budget_limit']:,} грн
* **Початковий очікуваний ризик:** {initial_risk:,.0f} грн
* **Кількість проаналізованих загроз:** {len(profile['threats'])}

## 2. Результати оптимізації
* **Фактичні витрати на захист:** {res['cost']:,.0f} грн
* **Залишковий ризик:** {res['new_risk']:,.0f} грн
* **Зниження ризику:** {res['reduction_percent']:.1f}% ({res['reduction']:,.0f} грн)
* **Показник ROI (Return on Investment):** {res['roi']:.2f}:1

## 3. Рекомендований розподіл бюджету по загрозах
"""
        if res['type'] == 'brute':
            for idx, t in enumerate(profile['threats']):
                c = profile['protection_costs'][idx + 1][res['combination'][idx]] if res['combination'][
                                                                                         idx] != '0%' else 0
                report_md += f"- **{t}**: Рівень захисту {res['combination'][idx]}, Витрати: {c:,} грн\n"
        else:
            for idx, t in enumerate(profile['threats']):
                report_md += f"- **{t}**: Витрати: {res['spending'][idx]:,.0f} грн\n"

        report_md += """
## 4. Висновки та обґрунтування
На основі проведеного математичного моделювання та аналізу можна зробити висновок, що запропонований розподіл коштів є найбільш ефективним та збалансованим у межах заданого бюджетного обмеження.
Застосування обраних засобів захисту дозволяє суттєво знизити ймовірність успішних кібератак, мінімізувати потенційні фінансові втрати та забезпечити безперебійну роботу організації.
Високий показник ROI повністю підтверджує економічну доцільність впровадження цих заходів безпеки, перетворюючи витрати на кібербезпеку в ефективну інвестицію.
"""
        st.markdown(report_md)
        st.download_button(
            label="📥 Завантажити звіт (Markdown)",
            data=report_md,
            file_name=f"CyberSecurity_Report_{org_type.replace(' ', '_')}.md",
            mime="text/markdown"
        )
    else:
        st.info("💡 Спочатку виконайте розрахунок у вкладці 'Оптимізація', щоб згенерувати детальний звіт.")

st.markdown("---")
