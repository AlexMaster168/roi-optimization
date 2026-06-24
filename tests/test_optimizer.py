import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from core.optimizer import BudgetOptimizer
from core.data_generator import CyberDataGenerator


class TestBudgetOptimizer:
    @pytest.fixture
    def optimizer(self):
        return BudgetOptimizer()

    @pytest.fixture
    def ecommerce_data(self):
        dg = CyberDataGenerator()
        p = dg.get_profile("E-commerce")
        return {
            "probs": np.array(p["base_probs"]),
            "losses": np.array(p["losses"]),
            "costs": p["protection_costs"],
            "budget": p["budget_limit"],
            "alpha": p["alpha"],
        }

    def test_calculate_risk(self, optimizer, ecommerce_data):
        risk = optimizer.calculate_risk(ecommerce_data["probs"], ecommerce_data["losses"])
        assert risk > 0

    def test_calculate_roi(self, optimizer):
        roi = optimizer.calculate_roi(1000, 500, 200)
        assert roi == 2.5

    def test_calculate_roi_zero_cost(self, optimizer):
        roi = optimizer.calculate_roi(1000, 500, 0)
        assert roi == 0

    def test_brute_force(self, optimizer, ecommerce_data):
        result = optimizer.optimize_brute_force(
            ecommerce_data["probs"], ecommerce_data["losses"],
            ecommerce_data["costs"], ecommerce_data["budget"],
        )
        assert result["best_solution"] is not None
        assert result["total_solutions"] > 0
        assert result["best_solution"]["cost"] <= ecommerce_data["budget"]
        assert result["best_solution"]["reduction"] >= 0

    def test_brute_force_with_alpha(self, optimizer, ecommerce_data):
        result = optimizer.optimize_brute_force(
            ecommerce_data["probs"], ecommerce_data["losses"],
            ecommerce_data["costs"], ecommerce_data["budget"],
            alpha=0.1,
        )
        assert result["best_solution"] is not None

    def test_continuous(self, optimizer, ecommerce_data):
        result = optimizer.optimize_continuous(
            ecommerce_data["probs"], ecommerce_data["losses"],
            ecommerce_data["budget"],
        )
        assert result["total_cost"] <= ecommerce_data["budget"]
        assert result["reduction"] >= 0
        assert result["roi"] >= 0

    def test_continuous_with_alpha(self, optimizer, ecommerce_data):
        result = optimizer.optimize_continuous(
            ecommerce_data["probs"], ecommerce_data["losses"],
            ecommerce_data["budget"], alpha=0.1,
        )
        assert result["total_cost"] <= ecommerce_data["budget"]

    def test_sensitivity_analysis(self, optimizer, ecommerce_data):
        results = optimizer.sensitivity_analysis(
            ecommerce_data["probs"], ecommerce_data["losses"],
            ecommerce_data["costs"], ecommerce_data["budget"],
            steps=5,
        )
        assert len(results) > 0
        assert all("budget" in r for r in results)
        assert all("reduction" in r for r in results)

    def test_pareto_front(self, optimizer, ecommerce_data):
        front = optimizer.pareto_front(
            ecommerce_data["probs"], ecommerce_data["losses"],
            ecommerce_data["costs"], ecommerce_data["budget"],
            n_points=10,
        )
        assert len(front) > 0
        assert front[0]["cost"] <= front[-1]["cost"]

    def test_compare_methods(self, optimizer, ecommerce_data):
        result = optimizer.compare_methods(
            ecommerce_data["probs"], ecommerce_data["losses"],
            ecommerce_data["costs"], ecommerce_data["budget"],
        )
        assert "brute_force" in result
        assert "continuous" in result

    def test_estimate_efficiency_factor(self, optimizer, ecommerce_data):
        ef = optimizer._estimate_efficiency_factor(
            ecommerce_data["probs"], ecommerce_data["losses"],
        )
        assert ef > 0
        assert ef < 1000000

    def test_all_org_types(self):
        dg = CyberDataGenerator()
        opt = BudgetOptimizer()
        for org in ["E-commerce", "Bank", "Industry", "Healthcare", "Telecom", "University"]:
            p = dg.get_profile(org)
            probs = np.array(p["base_probs"])
            losses = np.array(p["losses"])
            result = opt.optimize_brute_force(probs, losses, p["protection_costs"], p["budget_limit"])
            assert result["best_solution"] is not None, f"Failed for {org}"
