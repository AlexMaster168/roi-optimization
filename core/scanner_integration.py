import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from xml.etree import ElementTree

import numpy as np
import pandas as pd

from core.config import SEVERITY_LEVELS, PENTEST_PROB_ADJUSTMENTS

logger = logging.getLogger(__name__)


@dataclass
class Finding:
    name: str
    severity: str
    cwe: Optional[str] = None
    cvss: Optional[float] = None
    source_tool: str = ""
    description: str = ""
    remediation: str = ""
    affected_url: str = ""
    method: str = ""
    mitre_technique: str = ""
    mitre_tactic: str = ""

    @property
    def risk_score(self) -> float:
        return self.cvss if self.cvss else SEVERITY_LEVELS.get(self.severity, {}).get("cvss_range", (0, 0))[1]

    def to_dict(self) -> dict:
        return asdict(self)


class PentestReport:
    def __init__(self, findings: list[Finding]):
        self.findings = findings
        self.severity_counts = self._count_by_severity()
        self.total_risk = sum(f.risk_score for f in findings)

    def _count_by_severity(self) -> dict[str, int]:
        counts = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    def get_top_findings(self, n: int = 10) -> list[Finding]:
        return sorted(self.findings, key=lambda f: f.risk_score, reverse=True)[:n]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame()
        return pd.DataFrame([f.to_dict() for f in self.findings])


class OwaspReportParser:
    def __init__(self, mitre_map_path: Optional[str] = None):
        self.severity_multipliers = {
            level: data["multiplier"] for level, data in SEVERITY_LEVELS.items()
        }
        self.mitre_map = {}
        if mitre_map_path and os.path.exists(mitre_map_path):
            try:
                with open(mitre_map_path, "r", encoding="utf-8") as f:
                    self.mitre_map = json.load(f)
            except Exception as e:
                logger.warning("Failed to load MITRE mappings: %s", e)

    def _map_mitre(self, cwe_id: str) -> tuple[str, str]:
        cwe_key = f"CWE-{cwe_id}" if not str(cwe_id).startswith("CWE-") else str(cwe_id)
        entry = self.mitre_map.get(cwe_key, {})
        return entry.get("technique", ""), entry.get("tactic", "")

    def _severity_from_cvss(self, cvss: float) -> str:
        for level, data in SEVERITY_LEVELS.items():
            lo, hi = data["cvss_range"]
            if lo <= cvss <= hi and level != "Informational":
                return level
        if cvss >= 9.0:
            return "Critical"
        if cvss >= 7.0:
            return "High"
        if cvss >= 4.0:
            return "Medium"
        if cvss > 0:
            return "Low"
        return "Informational"

    def _map_severity(self, riskdesc: str) -> str:
        first_word = riskdesc.split("(")[0].strip().split(" ")[0].strip()
        for level in SEVERITY_LEVELS:
            if first_word.lower() == level.lower():
                return level
        return "Informational"

    def parse_zap_json(self, file_content: str) -> pd.DataFrame:
        findings = []
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse ZAP JSON: %s", e)
            return pd.DataFrame(columns=["name", "risk", "cwe", "severity", "cvss"])

        for site in data.get("site", []):
            for alert in site.get("alerts", []):
                severity = self._map_severity(alert.get("riskdesc", ""))
                cwe_id = alert.get("cweid", "")
                cvss_val = None
                cvss_data = alert.get("cvss", {})
                if isinstance(cvss_data, dict):
                    cvss_val = cvss_data.get("score")
                elif isinstance(cvss_data, (int, float)):
                    cvss_val = float(cvss_data)

                findings.append({
                    "name": alert.get("name", ""),
                    "risk": severity,
                    "cwe": str(cwe_id),
                    "severity": severity,
                    "cvss": cvss_val,
                    "source_tool": "OWASP ZAP",
                    "description": self._strip_html(alert.get("desc", "")),
                    "remediation": self._strip_html(alert.get("solution", "")),
                    "affected_url": self._extract_first_url(alert.get("instances", [])),
                    "method": self._extract_first_method(alert.get("instances", [])),
                })

        return pd.DataFrame(findings)

    def parse_zap_findings(self, file_content: str) -> list[Finding]:
        data = json.loads(file_content)
        findings = []
        for site in data.get("site", []):
            for alert in site.get("alerts", []):
                severity = self._map_severity(alert.get("riskdesc", ""))
                cwe_id = alert.get("cweid", "")
                mitre_tech, mitre_tactic = self._map_mitre(cwe_id)
                cvss_val = None
                cvss_data = alert.get("cvss", {})
                if isinstance(cvss_data, dict):
                    cvss_val = cvss_data.get("score")
                elif isinstance(cvss_data, (int, float)):
                    cvss_val = float(cvss_data)

                findings.append(Finding(
                    name=alert.get("name", ""),
                    severity=severity,
                    cwe=str(cwe_id),
                    cvss=cvss_val,
                    source_tool="OWASP ZAP",
                    description=self._strip_html(alert.get("desc", "")),
                    remediation=self._strip_html(alert.get("solution", "")),
                    affected_url=self._extract_first_url(alert.get("instances", [])),
                    method=self._extract_first_method(alert.get("instances", [])),
                    mitre_technique=mitre_tech,
                    mitre_tactic=mitre_tactic,
                ))
        return findings

    def parse_nikto_json(self, file_content: str) -> list[Finding]:
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Nikto JSON: %s", e)
            return []

        findings = []
        vulns = data.get("vulnerabilities", [])
        if not vulns and isinstance(data, list):
            vulns = data

        for vuln in vulns:
            msg = vuln.get("msg", vuln.get("description", ""))
            osvdb = vuln.get("osvdbid", vuln.get("id", ""))
            url = vuln.get("url", vuln.get("uri", ""))
            method = vuln.get("method", "GET")
            severity = self._nikto_severity(osvdb, msg)

            findings.append(Finding(
                name=f"Nikto: {msg[:100]}",
                severity=severity,
                cwe=str(osvdb),
                source_tool="Nikto",
                description=msg,
                remediation="Review and patch identified vulnerability.",
                affected_url=url,
                method=method,
            ))
        return findings

    def parse_nuclei_jsonl(self, file_content: str) -> list[Finding]:
        findings = []
        for line in file_content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            severity = entry.get("info", {}).get("severity", "info").capitalize()
            if severity not in SEVERITY_LEVELS:
                severity = "Informational"

            classification = entry.get("info", {}).get("classification", {})
            cwe_id = classification.get("cwe-id", classification.get("cwe", ""))
            cvss_val = classification.get("cvss-score", classification.get("cvss"))
            if isinstance(cvss_val, str):
                try:
                    cvss_val = float(cvss_val)
                except ValueError:
                    cvss_val = None

            mitre_tech, mitre_tactic = self._map_mitre(str(cwe_id))

            findings.append(Finding(
                name=entry.get("info", {}).get("name", entry.get("template-id", "Unknown")),
                severity=severity,
                cwe=str(cwe_id),
                cvss=cvss_val,
                source_tool="Nuclei",
                description=entry.get("info", {}).get("description", ""),
                remediation=entry.get("info", {}).get("remediation", ""),
                affected_url=entry.get("matched-at", entry.get("host", "")),
                mitre_technique=mitre_tech,
                mitre_tactic=mitre_tactic,
            ))
        return findings

    def parse_nmap_xml(self, file_content: str) -> list[Finding]:
        findings = []
        try:
            root = ElementTree.fromstring(file_content)
        except ElementTree.ParseError as e:
            logger.error("Failed to parse Nmap XML: %s", e)
            return []

        for host in root.findall(".//host"):
            addr_el = host.find("address")
            ip = addr_el.get("addr", "unknown") if addr_el is not None else "unknown"
            ports_el = host.find("ports")
            if ports_el is None:
                continue

            for port_el in ports_el.findall("port"):
                port_id = port_el.get("portid", "?")
                protocol = port_el.get("protocol", "?")
                state_el = port_el.find("state")
                state = state_el.get("state", "unknown") if state_el is not None else "unknown"
                if state != "open":
                    continue

                service_el = port_el.find("service")
                service_name = service_el.get("name", "unknown") if service_el is not None else "unknown"
                service_version = service_el.get("version", "") if service_el is not None else ""
                product = service_el.get("product", "") if service_el is not None else ""

                severity = self._nmap_service_severity(service_name, product, service_version)
                desc = f"Open port {port_id}/{protocol}: {service_name}"
                if product:
                    desc += f" ({product} {service_version})".strip()

                findings.append(Finding(
                    name=f"Nmap: {service_name} on {ip}:{port_id}",
                    severity=severity,
                    source_tool="Nmap",
                    description=desc,
                    affected_url=f"{ip}:{port_id}/{protocol}",
                ))
        return findings

    def parse_uploaded_report(self, file_content: str, format_hint: str = "auto") -> list[Finding]:
        if format_hint == "auto":
            format_hint = self._detect_format(file_content)

        parsers = {
            "zap_json": self.parse_zap_findings,
            "nikto_json": self.parse_nikto_json,
            "nuclei_jsonl": self.parse_nuclei_jsonl,
            "nmap_xml": self.parse_nmap_xml,
        }
        parser = parsers.get(format_hint, self._parse_generic_json)
        return parser(file_content)

    def adjust_probabilities(self, base_probs, threats, scan_report_df):
        adjusted_probs = np.array(base_probs, dtype=float)
        if scan_report_df.empty:
            return adjusted_probs.tolist()

        for i, threat in enumerate(threats):
            threat_lower = threat.lower()
            relevant = scan_report_df[
                scan_report_df["name"].str.contains(threat_lower, case=False, na=False)
            ]
            if relevant.empty:
                relevant = scan_report_df[
                    scan_report_df["name"].apply(lambda x: any(w in str(x).lower() for w in threat_lower.split()))
                ]
            if not relevant.empty:
                highest_risk = relevant["severity"].value_counts().idxmax() if "severity" in relevant.columns else relevant["risk"].iloc[0]
                multiplier = self.severity_multipliers.get(highest_risk, 1.0)
                adjusted_probs[i] = min(0.99, adjusted_probs[i] * multiplier)

        return adjusted_probs.tolist()

    def adjust_probs_from_findings(self, base_probs: list[float], findings: list[Finding]) -> list[float]:
        adjusted = np.array(base_probs, dtype=float).copy()
        if not findings:
            return adjusted.tolist()

        severity_max = {}
        for f in findings:
            cur = severity_max.get(f.severity, 0)
            severity_max[f.severity] = max(cur, f.risk_score)

        for sev, score in severity_max.items():
            adj = PENTEST_PROB_ADJUSTMENTS.get(sev, 0.0)
            if score >= 9.0:
                adj = max(adj, 0.15)
            elif score >= 7.0:
                adj = max(adj, 0.10)

        total_adj = sum(PENTEST_PROB_ADJUSTMENTS.get(s, 0) for s in severity_max)
        scale = min(total_adj / max(len(adjusted), 1), 0.3)

        for i in range(len(adjusted)):
            worst_severity = "Informational"
            for f in findings:
                if f.risk_score > SEVERITY_LEVELS.get(worst_severity, {}).get("cvss_range", (0, 0))[1]:
                    worst_severity = f.severity
            adj = PENTEST_PROB_ADJUSTMENTS.get(worst_severity, 0)
            adjusted[i] = min(0.99, adjusted[i] + adj)

        return adjusted.tolist()

    def generate_report(self, findings: list[Finding]) -> PentestReport:
        return PentestReport(findings)

    def _detect_format(self, content: str) -> str:
        content_stripped = content.strip()
        if content_stripped.startswith("<"):
            if "<nmaprun" in content_stripped or "<host>" in content_stripped:
                return "nmap_xml"
        if content_stripped.startswith("{"):
            try:
                data = json.loads(content_stripped)
                if "site" in data:
                    return "zap_json"
                if "vulnerabilities" in data or "host" in data:
                    return "nikto_json"
            except json.JSONDecodeError:
                pass
        if "\n" in content_stripped:
            first_line = content_stripped.split("\n")[0].strip()
            try:
                entry = json.loads(first_line)
                if "template-id" in entry or "info" in entry:
                    return "nuclei_jsonl"
            except json.JSONDecodeError:
                pass
        return "generic_json"

    def _parse_generic_json(self, file_content: str) -> list[Finding]:
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError:
            return []

        findings = []
        items = data if isinstance(data, list) else data.get("findings", data.get("results", []))
        if not isinstance(items, list):
            items = [items]

        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name", item.get("title", item.get("alert", "Unknown Finding")))
            severity = item.get("severity", item.get("risk", item.get("level", "Informational")))
            if severity.capitalize() not in SEVERITY_LEVELS:
                severity = "Informational"
            else:
                severity = severity.capitalize()

            findings.append(Finding(
                name=name,
                severity=severity,
                cwe=str(item.get("cwe", item.get("cweid", ""))),
                cvss=item.get("cvss"),
                source_tool=item.get("tool", "Unknown"),
                description=item.get("description", item.get("desc", "")),
                remediation=item.get("remediation", item.get("solution", "")),
                affected_url=item.get("url", item.get("uri", "")),
            ))
        return findings

    def _strip_html(self, text: str) -> str:
        import re
        clean = re.sub(r"<[^>]+>", "", text)
        return clean.strip()

    def _extract_first_url(self, instances: list) -> str:
        if instances and isinstance(instances, list) and len(instances) > 0:
            return instances[0].get("uri", "")
        return ""

    def _extract_first_method(self, instances: list) -> str:
        if instances and isinstance(instances, list) and len(instances) > 0:
            return instances[0].get("method", "")
        return ""

    def _nikto_severity(self, osvdb_id, msg: str) -> str:
        msg_lower = msg.lower()
        if any(w in msg_lower for w in ["remote code", "rce", "command injection", "sql injection"]):
            return "Critical"
        if any(w in msg_lower for w in ["xss", "cross-site", "directory traversal", "file include"]):
            return "High"
        if any(w in msg_lower for w in ["misconfiguration", "default password", "outdated", "header"]):
            return "Medium"
        if any(w in msg_lower for w in ["info", "disclosure", "cookie"]):
            return "Low"
        return "Informational"

    def _nmap_service_severity(self, service: str, product: str, version: str) -> str:
        dangerous = ["telnet", "ftp", "rlogin", "rsh"]
        if service.lower() in dangerous:
            return "High"
        if any(v in product.lower() for v in ["apache/2.2", "openssl/1.0", "openssh/6"]):
            return "Medium"
        if service.lower() in ["http", "https", "ssh", "dns"]:
            return "Informational"
        return "Low"
