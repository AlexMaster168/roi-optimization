import numpy as np
from scipy.optimize import minimize
from itertools import product


class BudgetOptimizer:
    def __init__(self):
        self.protection_levels = ['0%', '10%', '30%', '80%']
        self.reductions = {'0%': 0, '10%': 0.1, '30%': 0.3, '80%': 0.8}

    def calculate_risk(self, probs, losses):
        return np.sum(probs * losses)

    def calculate_roi(self, initial_risk, residual_risk, cost):
        if cost == 0:
            return 0
        avoided_loss = initial_risk - residual_risk
        return avoided_loss / cost

    def optimize_brute_force(self, probs, losses, protection_costs, budget_limit):
        n_threats = len(probs)
        best_solution = None
        best_reduction = 0
        all_solutions = []

        for combination in product(self.protection_levels, repeat=n_threats):
            total_cost = 0
            new_probs = probs.copy()

            for i in range(n_threats):
                level = combination[i]
                if level != '0%':
                    total_cost += protection_costs[i + 1][level]
                    new_probs[i] *= (1 - self.reductions[level])

            if total_cost <= budget_limit:
                initial_risk = self.calculate_risk(probs, losses)
                new_risk = self.calculate_risk(new_probs, losses)
                reduction = initial_risk - new_risk

                solution = {
                    'combination': combination,
                    'cost': total_cost,
                    'new_risk': new_risk,
                    'reduction': reduction,
                    'reduction_percent': (reduction / initial_risk * 100) if initial_risk > 0 else 0,
                    'roi': self.calculate_roi(initial_risk, new_risk, total_cost)
                }

                all_solutions.append(solution)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_solution = solution

        all_solutions.sort(key=lambda x: x['reduction'], reverse=True)

        return {
            'best_solution': best_solution,
            'top_solutions': all_solutions[:10],
            'total_solutions': len(all_solutions)
        }

    def optimize_continuous(self, probs, losses, budget_limit, efficiency_factor=50000):
        n_threats = len(probs)
        initial_risk = self.calculate_risk(probs, losses)

        def objective(spending):
            if np.sum(spending) > budget_limit:
                return 1e9

            k_coeffs = 1 - np.exp(-spending / efficiency_factor)
            new_probs = probs * (1 - k_coeffs)
            residual_risk = self.calculate_risk(new_probs, losses)

            return residual_risk + 0.1 * np.sum(spending)

        constraints = ({'type': 'ineq', 'fun': lambda x: budget_limit - np.sum(x)})
        bounds = [(0, budget_limit) for _ in range(n_threats)]
        x0 = np.ones(n_threats) * (budget_limit / n_threats)

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        optimal_spending = result.x
        k_coeffs = 1 - np.exp(-optimal_spending / efficiency_factor)
        new_probs = probs * (1 - k_coeffs)
        residual_risk = self.calculate_risk(new_probs, losses)

        return {
            'spending': optimal_spending,
            'total_cost': np.sum(optimal_spending),
            'residual_risk': residual_risk,
            'reduction': initial_risk - residual_risk,
            'reduction_percent': ((initial_risk - residual_risk) / initial_risk * 100) if initial_risk > 0 else 0,
            'roi': self.calculate_roi(initial_risk, residual_risk, np.sum(optimal_spending))
        }
