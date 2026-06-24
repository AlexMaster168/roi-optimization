import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pytest

from core.scanner_integration import OwaspReportParser, Finding, PentestReport


SAMPLE_ZAP_JSON = json.dumps({
    "site": [{
        "alerts": [
            {
                "name": "SQL Injection",
                "riskdesc": "High (Medium)",
                "cweid": "89",
                "desc": "<p>SQL injection possible</p>",
                "solution": "<p>Use parameterized queries</p>",
                "instances": [{"uri": "http://example.com/login", "method": "POST"}],
            },
            {
                "name": "Security Misconfiguration",
                "riskdesc": "Medium (Low)",
                "cweid": "693",
                "desc": "<p>Missing header</p>",
                "solution": "<p>Add header</p>",
                "instances": [{"uri": "http://example.com/", "method": "GET"}],
            },
        ]
    }]
})

SAMPLE_NIKTO_JSON = json.dumps({
    "host": "192.168.1.1",
    "ipport": "192.168.1.1:80",
    "vulnerabilities": [
        {"id": "1", "method": "GET", "url": "/admin", "msg": "Admin directory found"},
        {"id": "2", "method": "GET", "url": "/test", "msg": "SQL injection possible"},
    ],
})

SAMPLE_NUCLEI_JSONL = "\n".join([
    json.dumps({
        "template-id": "sqli-error-based",
        "info": {
            "name": "SQL Injection",
            "severity": "high",
            "description": "Error-based SQL injection",
            "classification": {"cwe-id": "CWE-89", "cvss-score": 8.6},
        },
        "matched-at": "http://example.com/login",
        "host": "http://example.com",
    }),
    json.dumps({
        "template-id": "missing-header",
        "info": {
            "name": "Missing Security Header",
            "severity": "low",
            "description": "X-Content-Type-Options missing",
            "classification": {"cwe-id": "CWE-693", "cvss-score": 2.0},
        },
        "matched-at": "http://example.com/",
    }),
])


class TestOwaspReportParser:
    @pytest.fixture
    def parser(self):
        mitre_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mitre_mappings.json")
        return OwaspReportParser(mitre_map_path=mitre_path)

    def test_parse_zap_json(self, parser):
        df = parser.parse_zap_json(SAMPLE_ZAP_JSON)
        assert len(df) == 2
        assert "name" in df.columns
        assert "risk" in df.columns

    def test_parse_zap_findings(self, parser):
        findings = parser.parse_zap_findings(SAMPLE_ZAP_JSON)
        assert len(findings) == 2
        assert isinstance(findings[0], Finding)
        assert findings[0].name == "SQL Injection"
        assert findings[0].severity == "High"

    def test_parse_nikto_json(self, parser):
        findings = parser.parse_nikto_json(SAMPLE_NIKTO_JSON)
        assert len(findings) == 2
        assert findings[0].source_tool == "Nikto"

    def test_parse_nuclei_jsonl(self, parser):
        findings = parser.parse_nuclei_jsonl(SAMPLE_NUCLEI_JSONL)
        assert len(findings) == 2
        assert findings[0].cvss == 8.6

    def test_detect_format_zap(self, parser):
        fmt = parser._detect_format(SAMPLE_ZAP_JSON)
        assert fmt == "zap_json"

    def test_detect_format_nikto(self, parser):
        fmt = parser._detect_format(SAMPLE_NIKTO_JSON)
        assert fmt == "nikto_json"

    def test_detect_format_nuclei(self, parser):
        fmt = parser._detect_format(SAMPLE_NUCLEI_JSONL)
        assert fmt == "nuclei_jsonl"

    def test_adjust_probabilities_empty(self, parser):
        import pandas as pd
        base = [0.1, 0.2, 0.3]
        result = parser.adjust_probabilities(base, ["A", "B", "C"], pd.DataFrame())
        assert result == [0.1, 0.2, 0.3]

    def test_adjust_probabilities_with_findings(self, parser):
        import pandas as pd
        base = [0.1, 0.2, 0.3]
        df = pd.DataFrame({
            "name": ["SQL Injection test"],
            "risk": ["High"],
            "cwe": ["89"],
            "severity": ["High"],
        })
        result = parser.adjust_probabilities(base, ["SQL Injection", "B", "C"], df)
        assert result[0] > 0.1
        assert result[1] == 0.2

    def test_map_mitre(self, parser):
        tech, tactic = parser._map_mitre("89")
        assert tech == "T1190"
        assert tactic == "Initial Access"

    def test_map_severity(self, parser):
        assert parser._map_severity("High (Medium)") == "High"
        assert parser._map_severity("Medium (Low)") == "Medium"
        assert parser._map_severity("Low (Low)") == "Low"

    def test_strip_html(self, parser):
        result = parser._strip_html("<p>Test <b>bold</b></p>")
        assert result == "Test bold"


class TestFinding:
    def test_finding_creation(self):
        f = Finding(name="Test", severity="High", cwe="89", cvss=8.5, source_tool="Nmap")
        assert f.name == "Test"
        assert f.risk_score == 8.5

    def test_finding_to_dict(self):
        f = Finding(name="Test", severity="Medium")
        d = f.to_dict()
        assert "name" in d
        assert d["severity"] == "Medium"

    def test_finding_risk_score_default(self):
        f = Finding(name="Test", severity="High")
        assert f.risk_score == 8.9


class TestPentestReport:
    def test_report_creation(self):
        findings = [
            Finding(name="A", severity="High"),
            Finding(name="B", severity="Medium"),
            Finding(name="C", severity="High"),
        ]
        report = PentestReport(findings)
        assert len(report.findings) == 3
        assert report.severity_counts["High"] == 2
        assert report.severity_counts["Medium"] == 1

    def test_top_findings(self):
        findings = [
            Finding(name="A", severity="Low", cvss=2.0),
            Finding(name="B", severity="High", cvss=8.5),
            Finding(name="C", severity="Critical", cvss=9.5),
        ]
        report = PentestReport(findings)
        top = report.get_top_findings(2)
        assert top[0].name == "C"
        assert top[1].name == "B"

    def test_to_dataframe(self):
        findings = [Finding(name="A", severity="High")]
        report = PentestReport(findings)
        df = report.to_dataframe()
        assert len(df) == 1
        assert "name" in df.columns
