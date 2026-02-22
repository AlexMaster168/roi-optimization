import json
import pandas as pd
import numpy as np


class OwaspReportParser:
    def __init__(self):
        self.severity_multipliers = {
            'High': 1.5,
            'Medium': 1.2,
            'Low': 1.05,
            'Informational': 1.0
        }

    def parse_zap_json(self, file_content):
        data = json.loads(file_content)
        findings = []
        for site in data.get('site', []):
            for alert in site.get('alerts', []):
                findings.append({
                    'name': alert.get('name'),
                    'risk': alert.get('riskdesc', '').split(' ')[0],
                    'cwe': alert.get('cweid')
                })
        return pd.DataFrame(findings)

    def adjust_probabilities(self, base_probs, threats, scan_report_df):
        adjusted_probs = np.array(base_probs, dtype=float)
        if scan_report_df.empty:
            return adjusted_probs.tolist()

        for i, threat in enumerate(threats):
            relevant_findings = scan_report_df[
                scan_report_df['name'].str.contains(threat.split(' ')[0], case=False, na=False)]
            if not relevant_findings.empty:
                highest_risk = relevant_findings['risk'].iloc[0]
                multiplier = self.severity_multipliers.get(highest_risk, 1.0)
                adjusted_probs[i] = min(0.99, adjusted_probs[i] * multiplier)

        return adjusted_probs.tolist()
