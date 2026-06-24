import streamlit as st

from core.config import DEFAULT_ORG_TYPES

st.set_page_config(page_title="Звіт", page_icon="📑", layout="wide")

st.header("📑 Детальний звіт та висновки")

org_type = st.session_state.get("selected_org", DEFAULT_ORG_TYPES[0])
use_ml = st.session_state.get("use_ml", False)

if "last_optimization_result" in st.session_state:
    res = st.session_state["last_optimization_result"]

    from core.data_generator import CyberDataGenerator
    dg = CyberDataGenerator()
    profile = dg.get_profile(org_type)

    report_md = f"""# Детальний звіт з оптимізації кібербезпеки

**Організація:** {org_type}
**Використання ML-прогнозування:** {'Так' if use_ml else 'Ні'}
**Метод оптимізації:** {res['type']}

## 1. Початкові показники
* **Загальний ліміт бюджету:** {profile['budget_limit']:,} грн
* **Кількість проаналізованих загроз:** {len(profile['threats'])}

## 2. Результати оптимізації
* **Фактичні витрати на захист:** {res['cost']:,.0f} грн
* **Залишковий ризик:** {res['new_risk']:,.0f} грн
* **Зниження ризику:** {res['reduction_percent']:.1f}% ({res['reduction']:,.0f} грн)
* **Показник ROI (Return on Investment):** {res['roi']:.2f}:1

## 3. Рекомендований розподіл бюджету по загрозах
"""

    if res["type"] == "brute":
        for idx, t in enumerate(profile["threats"]):
            c = profile["protection_costs"][idx + 1][res["combination"][idx]] if res["combination"][idx] != "0%" else 0
            report_md += f"- **{t}**: Рівень захисту {res['combination'][idx]}, Витрати: {c:,} грн\n"
    else:
        for idx, t in enumerate(profile["threats"]):
            report_md += f"- **{t}**: Витрати: {res['spending'][idx]:,.0f} грн\n"

    if "pentest_findings" in st.session_state and st.session_state["pentest_findings"]:
        findings = st.session_state["pentest_findings"]
        severity_counts = {}
        for f in findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        report_md += f"""
## 4. Результати пентестингу
* **Загальна кількість знахідок:** {len(findings)}
* **Критичних:** {severity_counts.get('Critical', 0)}
* **Високих:** {severity_counts.get('High', 0)}
* **Середніх:** {severity_counts.get('Medium', 0)}
* **Низьких:** {severity_counts.get('Low', 0)}
"""
        if "pentest_remediation" in st.session_state:
            report_md += "\n### Рекомендації з усунення\n\n"
            for item in st.session_state["pentest_remediation"][:10]:
                report_md += f"{item['order']}. **[{item['severity']}]** {item['finding']}\n"
                report_md += f"   {item['remediation']}\n\n"

    report_md += """
## 5. Висновки та обґрунтування
На основі проведеного математичного моделювання та аналізу можна зробити висновок, що запропонований розподіл коштів є найбільш ефективним та збалансованим у межах заданого бюджетного обмеження.
Застосування обраних засобів захисту дозволяє суттєво знизити ймовірність успішних кібератак, мінімізувати потенційні фінансові втрати та забезпечити безперебійну роботу організації.
Високий показник ROI повністю підтверджує економічну доцільність впровадження цих заходів безпеки.
"""
    st.markdown(report_md)

    st.download_button(
        label="📥 Завантажити звіт (Markdown)",
        data=report_md,
        file_name=f"CyberSecurity_Report_{org_type.replace(' ', '_')}.md",
        mime="text/markdown",
    )
else:
    st.info("💡 Спочатку виконайте розрахунок у вкладці 'Оптимізація', щоб згенерувати детальний звіт.")
