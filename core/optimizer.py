import numpy as np
from scipy.optimize import minimize
from itertools import product
import logging

from core.config import PROTECTION_LEVELS, PROTECTION_REDUCTIONS

logger = logging.getLogger(__name__)


class BudgetOptimizer:
    def __init__(self):
        self.protection_levels = PROTECTION_LEVELS
        self.reductions = PROTECTION_REDUCTIONS

    def calculate_risk(self, probs, losses):
        return np.sum(probs * losses)

    def calculate_roi(self, initial_risk, residual_risk, cost):
        if cost == 0:
            return 0
        avoided_loss = initial_risk - residual_risk
        return avoided_loss / cost

    def optimize_brute_force(self, probs, losses, protection_costs, budget_limit, alpha=0.0):
        n_threats = len(probs)
        best_solution = None
        best_score = -float("inf")
        all_solutions = []

        initial_risk = self.calculate_risk(probs, losses)

        for combination in product(self.protection_levels, repeat=n_threats):
            total_cost = 0
            new_probs = probs.copy()
            for i in range(n_threats):
                level = combination[i]
                if level != "0%":
                    total_cost += protection_costs[i + 1][level]
                    new_probs[i] *= (1 - self.reductions[level])

            if total_cost <= budget_limit:
                new_risk = self.calculate_risk(new_probs, losses)
                reduction = initial_risk - new_risk
                roi = self.calculate_roi(initial_risk, new_risk, total_cost)

                score = reduction - alpha * total_cost

                solution = {
                    "combination": combination,
                    "cost": total_cost,
                    "new_risk": new_risk,
                    "reduction": reduction,
                    "reduction_percent": (reduction / initial_risk * 100) if initial_risk > 0 else 0,
                    "roi": roi,
                    "score": score,
                }
                all_solutions.append(solution)
                if score > best_score:
                    best_score = score
                    best_solution = solution

        all_solutions.sort(key=lambda x: x["reduction"], reverse=True)
        return {
            "best_solution": best_solution,
            "top_solutions": all_solutions[:10],
            "total_solutions": len(all_solutions),
        }

    def optimize_continuous(self, probs, losses, budget_limit, efficiency_factor=None, alpha=0.0):
        n_threats = len(probs)
        initial_risk = self.calculate_risk(probs, losses)

        if efficiency_factor is None:
            efficiency_factor = self._estimate_efficiency_factor(probs, losses)

        def objective(spending):
            if np.sum(spending) > budget_limit:
                return 1e9
            k_coeffs = 1 - np.exp(-spending / efficiency_factor)
            new_probs = probs * (1 - k_coeffs)
            residual_risk = self.calculate_risk(new_probs, losses)
            return residual_risk + alpha * np.sum(spending)

        constraints = ({"type": "ineq", "fun": lambda x: budget_limit - np.sum(x)})
        bounds = [(0, budget_limit) for _ in range(n_threats)]
        x0 = np.ones(n_threats) * (budget_limit / n_threats)
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        optimal_spending = result.x
        k_coeffs = 1 - np.exp(-optimal_spending / efficiency_factor)
        new_probs = probs * (1 - k_coeffs)
        residual_risk = self.calculate_risk(new_probs, losses)

        return {
            "spending": optimal_spending,
            "total_cost": np.sum(optimal_spending),
            "residual_risk": residual_risk,
            "reduction": initial_risk - residual_risk,
            "reduction_percent": ((initial_risk - residual_risk) / initial_risk * 100) if initial_risk > 0 else 0,
            "roi": self.calculate_roi(initial_risk, residual_risk, np.sum(optimal_spending)),
            "efficiency_factor": efficiency_factor,
        }

    def _estimate_efficiency_factor(self, probs, losses):
        risk_weighted = probs * losses
        total_risk = np.sum(risk_weighted)
        if total_risk == 0:
            return 50000
        max_single = np.max(risk_weighted)
        return max(10000, min(200000, max_single * 0.1))

    def sensitivity_analysis(self, probs, losses, protection_costs, base_budget, alpha=0.0,
                             steps=11, budget_range=(0.5, 1.5)):
        budget_factors = np.linspace(budget_range[0], budget_range[1], steps)
        results = []

        for factor in budget_factors:
            budget = int(base_budget * factor)
            res = self.optimize_brute_force(probs, losses, protection_costs, budget, alpha)
            if res["best_solution"]:
                results.append({
                    "budget": budget,
                    "budget_factor": factor,
                    "cost": res["best_solution"]["cost"],
                    "reduction": res["best_solution"]["reduction"],
                    "reduction_percent": res["best_solution"]["reduction_percent"],
                    "roi": res["best_solution"]["roi"],
                    "new_risk": res["best_solution"]["new_risk"],
                })

        return results

    def pareto_front(self, probs, losses, protection_costs, budget_limit, alpha=0.0, n_points=50):
        budget_steps = np.linspace(0, budget_limit, n_points + 1)[1:]
        pareto_points = []

        for budget in budget_steps:
            res = self.optimize_brute_force(probs, losses, protection_costs, int(budget), alpha)
            if res["best_solution"]:
                pareto_points.append({
                    "cost": res["best_solution"]["cost"],
                    "risk_reduction": res["best_solution"]["reduction"],
                    "residual_risk": res["best_solution"]["new_risk"],
                    "roi": res["best_solution"]["roi"],
                })

        if not pareto_points:
            return []

        pareto_points.sort(key=lambda x: x["cost"])
        front = [pareto_points[0]]
        max_reduction = pareto_points[0]["risk_reduction"]

        for point in pareto_points[1:]:
            if point["risk_reduction"] > max_reduction:
                front.append(point)
                max_reduction = point["risk_reduction"]

        return front

    def compare_methods(self, probs, losses, protection_costs, budget_limit, alpha=0.0):
        bf = self.optimize_brute_force(probs, losses, protection_costs, budget_limit, alpha)
        cont = self.optimize_continuous(probs, losses, budget_limit, alpha=alpha)

        results = {"brute_force": bf, "continuous": cont}
        if bf["best_solution"]:
            results["brute_force_best"] = {
                "cost": bf["best_solution"]["cost"],
                "reduction": bf["best_solution"]["reduction"],
                "reduction_percent": bf["best_solution"]["reduction_percent"],
                "roi": bf["best_solution"]["roi"],
            }
        results["continuous_result"] = {
            "cost": cont["total_cost"],
            "reduction": cont["reduction"],
            "reduction_percent": cont["reduction_percent"],
            "roi": cont["roi"],
        }
        return results
