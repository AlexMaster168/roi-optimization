import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


class ThreatForecaster:
    def __init__(self, model_type='auto'):
        self.model_type = model_type
        self.models = []
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.best_model_names = []
        self.residuals_std = []

    def train(self, history_df, n_threats):
        self.models = []
        self.best_model_names = []
        self.residuals_std = []
        X = history_df['year'].values.reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)

        model_classes = {
            'linear': LinearRegression,
            'ridge': lambda: Ridge(alpha=1.0),
            'lasso': lambda: Lasso(alpha=0.01),
            'elastic_net': lambda: ElasticNet(alpha=0.01),
            'random_forest': lambda: RandomForestRegressor(n_estimators=50, random_state=42),
            'gradient_boosting': lambda: GradientBoostingRegressor(n_estimators=50, random_state=42),
            'svr': lambda: SVR(kernel='rbf', C=1.0, epsilon=0.01),
            'decision_tree': lambda: DecisionTreeRegressor(random_state=42)
        }

        for i in range(n_threats):
            y = np.array([p[i] for p in history_df['probs']])

            if self.model_type == 'auto':
                best_model = None
                best_r2 = -float('inf')
                best_name = ''

                for name, model_cls in model_classes.items():
                    temp_model = model_cls()
                    temp_model.fit(X_scaled, y)
                    y_pred = temp_model.predict(X_scaled)
                    r2 = r2_score(y, y_pred)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = temp_model
                        best_name = name

                self.models.append(best_model)
                self.best_model_names.append(best_name)
                residuals = y - best_model.predict(X_scaled)
                self.residuals_std.append(np.std(residuals) if np.std(residuals) > 0 else 0.01)
            else:
                model = model_classes.get(self.model_type, LinearRegression)()
                model.fit(X_scaled, y)
                self.models.append(model)
                self.best_model_names.append(self.model_type)
                residuals = y - model.predict(X_scaled)
                self.residuals_std.append(np.std(residuals) if np.std(residuals) > 0 else 0.01)

        self.is_fitted = True
        return self

    def predict(self, year_value):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_pred = np.array([[year_value]])
        X_scaled = self.scaler.transform(X_pred)
        predictions = [model.predict(X_scaled)[0] for model in self.models]
        return np.clip(predictions, 0.01, 0.99)

    def predict_monte_carlo(self, year_value, n_simulations=1000):
        base_preds = self.predict(year_value)
        simulations = []
        for i in range(len(self.models)):
            sims = np.random.normal(loc=base_preds[i], scale=self.residuals_std[i], size=n_simulations)
            sims = np.clip(sims, 0.01, 0.99)
            simulations.append(sims)
        return np.array(simulations)

    def get_model_metrics(self, history_df):
        X = history_df['year'].values.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)
        metrics = []
        for i, model in enumerate(self.models):
            y_true = np.array([p[i] for p in history_df['probs']])
            y_pred = model.predict(X_scaled)
            metrics.append({
                "threat_id": i + 1,
                "model": self.best_model_names[i],
                "mse": mean_squared_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred)
            })
        return pd.DataFrame(metrics)

    def compare_all_models(self, history_df, n_threats):
        X = history_df['year'].values.reshape(-1, 1)
        X_scaled = StandardScaler().fit_transform(X)
        model_classes = {
            'Linear': LinearRegression,
            'Ridge': lambda: Ridge(alpha=1.0),
            'Lasso': lambda: Lasso(alpha=0.01),
            'ElasticNet': lambda: ElasticNet(alpha=0.01),
            'Random Forest': lambda: RandomForestRegressor(n_estimators=50, random_state=42),
            'Gradient Boosting': lambda: GradientBoostingRegressor(n_estimators=50, random_state=42),
            'SVR': lambda: SVR(kernel='rbf', C=1.0, epsilon=0.01),
            'Decision Tree': lambda: DecisionTreeRegressor(random_state=42)
        }
        results = []
        for i in range(n_threats):
            y = np.array([p[i] for p in history_df['probs']])
            for name, model_cls in model_classes.items():
                model = model_cls()
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                results.append({
                    "Threat ID": i + 1,
                    "Model": name,
                    "MSE": mean_squared_error(y, y_pred),
                    "R2": r2_score(y, y_pred)
                })
        return pd.DataFrame(results)

    def plot_forecast(self, history_df, forecast_years=3, org_type="Organization", n_simulations=1000):
        fig = go.Figure()
        last_year = history_df['year'].max()
        forecast_x = list(range(last_year + 1, last_year + forecast_years + 1))

        for i in range(len(history_df['probs'].iloc[0])):
            hist_probs = [p[i] for p in history_df['probs']]
            fig.add_trace(go.Scatter(
                x=history_df['year'],
                y=hist_probs,
                mode='lines+markers',
                name=f'Загроза {i + 1}',
                line=dict(width=2)
            ))

            forecast_y = []
            lower_bound = []
            upper_bound = []

            for year in forecast_x:
                sims = self.predict_monte_carlo(year, n_simulations)[i]
                forecast_y.append(np.mean(sims))
                lower_bound.append(np.percentile(sims, 5))
                upper_bound.append(np.percentile(sims, 95))

            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode='lines+markers',
                name=f'Прогноз {i + 1} ({self.best_model_names[i]})',
                line=dict(dash='dash', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=forecast_x + forecast_x[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'ДІ 90% {i + 1}',
                showlegend=False
            ))

        fig.update_layout(
            title=f'Прогноз для {org_type} (Monte Carlo, N={n_simulations})',
            xaxis_title='Рік',
            yaxis_title='Ймовірність',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        return fig
