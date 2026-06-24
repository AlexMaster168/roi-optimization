import os
from datetime import datetime
from typing import Optional

from core.config import SEVERITY_COLORS


class ReportGenerator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def generate_pentest_report(self, pentest_report, org_name: str = "Organization") -> str:
        findings = pentest_report.findings
        severity_counts = pentest_report.severity_counts
        total_risk = pentest_report.total_risk

        md = f"""# Звіт пентестингу: {org_name}

**Дата:** {self.timestamp}
**Загальна кількість знахідок:** {len(findings)}
**Сукупний показник ризику:** {total_risk:.1f}

## Розподіл за тяжкістю

| Тяжкість | Кількість |
|----------|-----------|
"""
        for sev in ["Critical", "High", "Medium", "Low", "Informational"]:
            count = severity_counts.get(sev, 0)
            if count > 0:
                md += f"| {sev} | {count} |\n"

        md += "\n## Топ-10 знахідок\n\n"
        for f in pentest_report.get_top_findings(10):
            cwe_str = f"CWE-{f.cwe}" if f.cwe and not f.cwe.startswith("CWE-") else f.cwe or ""
            cvss_str = f"CVSS: {f.cvss:.1f}" if f.cvss else ""
            md += f"### [{f.severity}] {f.name}\n\n"
            if f.source_tool:
                md += f"**Джерело:** {f.source_tool}\n\n"
            if cvss_str or cwe_str:
                md += f"**Метрики:** {cvss_str} {cwe_str}\n\n"
            if f.affected_url:
                md += f"**URL/Адреса:** `{f.affected_url}`\n\n"
            if f.description:
                md += f"**Опис:** {f.description}\n\n"
            if f.remediation:
                md += f"**Рекомендація:** {f.remediation}\n\n"
            md += "---\n\n"

        return md

    def generate_budget_report(self, optimization_result, org_name: str = "Organization",
                               use_ml: bool = False, method: str = "Brute Force",
                               pentest_data: Optional[dict] = None) -> str:
        md = f"""# Детальний звіт з оптимізації кібербезпеки

**Організація:** {org_name}
**Використання ML-прогнозування:** {'Так' if use_ml else 'Ні'}
**Метод оптимізації:** {method}
**Дата:** {self.timestamp}

## 1. Результати оптимізації

* **Фактичні витрати на захист:** {optimization_result.get('cost', optimization_result.get('total_cost', 0)):,.0f} грн
* **Залишковий ризик:** {optimization_result.get('new_risk', optimization_result.get('residual_risk', 0)):,.0f} грн
* **Зниження ризику:** {optimization_result.get('reduction_percent', 0):.1f}% ({optimization_result.get('reduction', 0):,.0f} грн)
* **Показник ROI:** {optimization_result.get('roi', 0):.2f}:1
"""

        if pentest_data:
            md += f"""
## 2. Результати пентестингу

* **Кількість знахідок:** {pentest_data.get('total_findings', 0)}
* **Критичних:** {pentest_data.get('critical', 0)}
* **Високих:** {pentest_data.get('high', 0)}
* **Середніх:** {pentest_data.get('medium', 0)}
* **Низьких:** {pentest_data.get('low', 0)}
* **Сукупний показник ризику:** {pentest_data.get('risk_score', 0):.1f}
"""

            if pentest_data.get("remediation"):
                md += "\n### Рекомендації з усунення\n\n"
                for item in pentest_data["remediation"][:10]:
                    md += f"{item['order']}. **[{item['severity']}]** {item['finding']}\n"
                    md += f"   {item['remediation']}\n\n"

        md += """
## 3. Висновки

На основі проведеного аналізу запропонований розподіл коштів є оптимальним в межах заданого бюджетного обмеження.
Застосування обраних засобів захисту дозволяє суттєво знизити ймовірність успішних кібератак,
мінімізувати потенційні фінансові втрати та забезпечити безперебійну роботу організації.
"""
        return md

    def generate_combined_report(self, org_name: str, forecast_data: Optional[dict] = None,
                                  optimization_data: Optional[dict] = None,
                                  pentest_data: Optional[dict] = None) -> str:
        md = f"""# Комбінований звіт кібербезпеки: {org_name}

**Дата:** {self.timestamp}

---

"""
        if pentest_data and pentest_data.get("report"):
            md += self.generate_pentest_report(pentest_data["report"], org_name)
            md += "\n---\n\n"

        if forecast_data:
            md += self._format_forecast_section(forecast_data)
            md += "\n---\n\n"

        if optimization_data:
            pentest_section = pentest_data if pentest_data else None
            md += self.generate_budget_report(
                optimization_data, org_name,
                use_ml=forecast_data is not None,
                pentest_data=pentest_section,
            )

        return md

    def generate_remediation_plan(self, findings) -> list[dict]:
        plan = []
        severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Informational": 4}

        for f in findings:
            if not f.remediation:
                continue
            plan.append({
                "priority": severity_order.get(f.severity, 5),
                "finding": f.name,
                "severity": f.severity,
                "remediation": f.remediation,
                "source": f.source_tool,
                "cwe": f.cwe,
                "cvss": f.cvss,
            })

        plan.sort(key=lambda x: (x["priority"], -(x["cvss"] or 0)))
        for i, item in enumerate(plan):
            item["order"] = i + 1

        return plan

    def _format_forecast_section(self, forecast_data: dict) -> str:
        md = "## Прогноз загроз\n\n"
        if "metrics" in forecast_data:
            md += "### Метрики моделей\n\n"
            md += "| Загроза | Модель | MSE | R2 | CV R2 |\n"
            md += "|---------|--------|-----|----|-------|\n"
            for m in forecast_data["metrics"]:
                md += f"| {m.get('threat_id', '')} | {m.get('model', '')} | "
                md += f"{m.get('mse', 0):.6f} | {m.get('r2', 0):.4f} | {m.get('cv_r2', 0):.4f} |\n"

        if "forecast" in forecast_data:
            md += "\n### Прогнозовані ймовірності\n\n"
            md += "| Загроза | Поточна | Прогноз | Min (5%) | Max (95%) |\n"
            md += "|---------|---------|---------|----------|----------|\n"
            for f in forecast_data["forecast"]:
                md += f"| {f.get('threat', '')} | {f.get('current', 0):.3f} | "
                md += f"{f.get('mean', 0):.3f} | {f.get('low', 0):.3f} | {f.get('high', 0):.3f} |\n"

        return md
