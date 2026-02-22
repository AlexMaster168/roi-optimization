import numpy as np
import pandas as pd


class CyberDataGenerator:
    """Генератор історичних даних про кіберзагрози на основі статті"""

    def __init__(self):
        self.org_profiles = {
            "E-commerce": {
                "threats": ["DDoS", "Data Leak", "Virus", "Hardware Fail", "Human Error", "Phishing"],
                "base_probs": [0.24, 0.15, 0.147, 0.253, 0.204, 0.220],
                "losses": [50000, 2500000, 100000, 50000, 30000, 40000],
                "protection_costs": {
                    1: {'10%': 25000, '30%': 85000, '80%': 250000},
                    2: {'10%': 35000, '30%': 120000, '80%': 400000},
                    3: {'10%': 28000, '30%': 95000, '80%': 320000},
                    4: {'10%': 18000, '30%': 65000, '80%': 200000},
                    5: {'10%': 10000, '30%': 35000, '80%': 120000},
                    6: {'10%': 6000, '30%': 22000, '80%': 75000}
                },
                "budget_limit": 267000,
                "alpha": 0.15
            },
            "Bank": {
                "threats": ["APT", "Insider", "ABS Compromise", "Fraud", "Tech Fail", "Protocol"],
                "base_probs": [0.10, 0.15, 0.20, 0.25, 0.15, 0.15],
                "losses": [1500000, 500000, 1000000, 2000000, 100000, 100000],
                "protection_costs": {
                    1: {'10%': 95000, '30%': 320000, '80%': 950000},
                    2: {'10%': 75000, '30%': 250000, '80%': 750000},
                    3: {'10%': 60000, '30%': 200000, '80%': 600000},
                    4: {'10%': 42000, '30%': 140000, '80%': 420000},
                    5: {'10%': 22000, '30%': 75000, '80%': 220000},
                    6: {'10%': 12000, '30%': 40000, '80%': 120000}
                },
                "budget_limit": 624000,
                "alpha": 0.12
            },
            "Industry": {
                "threats": ["SCADA Attack", "Espionage", "Automation Fail", "Operator Error", "Tech Fail", "Protocol"],
                "base_probs": [0.041, 0.138, 0.182, 0.224, 0.197, 0.218],
                "losses": [5000000, 1000000, 2000000, 500000, 300000, 200000],
                "protection_costs": {
                    1: {'10%': 125000, '30%': 420000, '80%': 1250000},
                    2: {'10%': 88000, '30%': 290000, '80%': 880000},
                    3: {'10%': 68000, '30%': 225000, '80%': 680000},
                    4: {'10%': 48000, '30%': 160000, '80%': 480000},
                    5: {'10%': 32000, '30%': 105000, '80%': 320000},
                    6: {'10%': 20000, '30%': 65000, '80%': 200000}
                },
                "budget_limit": 700000,
                "alpha": 0.10
            },
            "Healthcare": {
                "threats": ["Ransomware", "Medical Data Leak", "System Downtime", "Equipment Fail", "Staff Error",
                            "Unauthorized Access"],
                "base_probs": [0.019, 0.141, 0.198, 0.204, 0.203, 0.235],
                "losses": [3000000, 2200000, 650000, 380000, 120000, 95000],
                "protection_costs": {
                    1: {'10%': 60000, '30%': 200000, '80%': 600000},
                    2: {'10%': 95000, '30%': 320000, '80%': 950000},
                    3: {'10%': 48000, '30%': 160000, '80%': 480000},
                    4: {'10%': 40000, '30%': 135000, '80%': 400000},
                    5: {'10%': 20000, '30%': 65000, '80%': 200000},
                    6: {'10%': 15000, '30%': 50000, '80%': 150000}
                },
                "budget_limit": 685000,
                "alpha": 0.14
            },
            "Telecom": {
                "threats": ["Cyber Attack", "Subscriber Data", "DDoS", "Equipment Fail", "Config Error",
                            "Physical Damage"],
                "base_probs": [0.031, 0.178, 0.203, 0.198, 0.181, 0.209],
                "losses": [2800000, 1400000, 750000, 450000, 220000, 180000],
                "protection_costs": {
                    1: {'10%': 115000, '30%': 380000, '80%': 1150000},
                    2: {'10%': 75000, '30%': 250000, '80%': 750000},
                    3: {'10%': 60000, '30%': 200000, '80%': 600000},
                    4: {'10%': 48000, '30%': 160000, '80%': 480000},
                    5: {'10%': 27000, '30%': 90000, '80%': 270000},
                    6: {'10%': 22000, '30%': 72000, '80%': 220000}
                },
                "budget_limit": 635000,
                "alpha": 0.11
            },
            "University": {
                "threats": ["Server Attack", "Data Leak", "Downtime", "Device Infection", "Unauthorized Use",
                            "Data Loss"],
                "base_probs": [0.052, 0.198, 0.201, 0.177, 0.162, 0.210],
                "losses": [850000, 1200000, 420000, 280000, 150000, 320000],
                "protection_costs": {
                    1: {'10%': 48000, '30%': 160000, '80%': 480000},
                    2: {'10%': 68000, '30%': 225000, '80%': 680000},
                    3: {'10%': 40000, '30%': 135000, '80%': 400000},
                    4: {'10%': 32000, '30%': 105000, '80%': 320000},
                    5: {'10%': 18000, '30%': 60000, '80%': 180000},
                    6: {'10%': 38000, '30%': 125000, '80%': 380000}
                },
                "budget_limit": 513000,
                "alpha": 0.16
            }
        }

    def generate_history(self, org_type, years=5, noise_level=0.02):
        """Генерує історичні дані за кілька років для навчання ML-моделі"""
        profile = self.org_profiles[org_type]
        base_probs = np.array(profile["base_probs"])
        data = []

        for year in range(years):
            # Додаємо тренд + шум до ймовірностей (імітація реальної динаміки)
            trend = np.linspace(0, 0.03 * year, len(base_probs))
            noise = np.random.normal(0, noise_level, size=len(base_probs))
            probs = np.clip(base_probs + noise + trend, 0.01, 0.99)

            # Розрахунок ризику за рік
            total_risk = np.sum(probs * np.array(profile["losses"]))

            data.append({
                "year": year,
                "probs": probs.tolist(),
                "total_risk": total_risk,
                "incident_count": int(np.sum(probs) * 1000)
            })

        return pd.DataFrame(data)

    def get_profile(self, org_type):
        return self.org_profiles[org_type]