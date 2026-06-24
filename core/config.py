import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

SCAN_DEFAULTS = {
    "nmap_args": "-sV -sC --open",
    "nmap_args_safe": "-sT --open",
    "nikto_timeout": 300,
    "nuclei_severity": "critical,high,medium",
    "scan_timeout": 600,
}

SEVERITY_LEVELS = {
    "Critical": {"multiplier": 1.8, "color": "#D32F2F", "cvss_range": (9.0, 10.0)},
    "High": {"multiplier": 1.5, "color": "#F57C00", "cvss_range": (7.0, 8.9)},
    "Medium": {"multiplier": 1.2, "color": "#FBC02D", "cvss_range": (4.0, 6.9)},
    "Low": {"multiplier": 1.05, "color": "#388E3C", "cvss_range": (0.1, 3.9)},
    "Informational": {"multiplier": 1.0, "color": "#1976D2", "cvss_range": (0.0, 0.0)},
}

SEVERITY_COLORS = {
    "Critical": "#D32F2F",
    "High": "#F57C00",
    "Medium": "#FBC02D",
    "Low": "#388E3C",
    "Informational": "#1976D2",
}

PROTECTION_LEVELS = ["0%", "10%", "30%", "80%"]
PROTECTION_REDUCTIONS = {"0%": 0, "10%": 0.1, "30%": 0.3, "80%": 0.8}

PENTEST_PROB_ADJUSTMENTS = {
    "Critical": 0.15,
    "High": 0.10,
    "Medium": 0.05,
    "Low": 0.02,
    "Informational": 0.0,
}

DEFAULT_ORG_TYPES = ["E-commerce", "Bank", "Industry", "Healthcare", "Telecom", "University"]

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = "INFO"
