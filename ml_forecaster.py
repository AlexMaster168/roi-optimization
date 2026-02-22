import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


class ThreatForecaster:
    """Прогнозування ймовірностей загроз за допомогою регресії"""

    def __init__(self, model_type='linear'):
        self.model_type = model_type
        self.models = []
        self.scaler = StandardScaler()
        self.is_fitted = False

    def train(self, history_df, n_threats):
        """Навчання моделі для кожної загрози окремо"""
        self.models = []
        X = history_df['year'].values.reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)

        for i in range(n_threats):
            y = np.array([p[i] for p in history_df['probs']])

            if self.model_type == 'linear':
                model = LinearRegression()
            elif self.model_type == 'ridge':
                model = Ridge(alpha=1.0)
            elif self.model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                model = LinearRegression()

            model.fit(X_scaled, y)
            self.models.append(model)

        self.is_fitted = True
        return self

    def predict(self, year_value):
        """Прогноз ймовірностей на вказаний рік"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        X_pred = np.array([[year_value]])
        X_scaled = self.scaler.transform(X_pred)

        predictions = [model.predict(X_scaled)[0] for model in self.models]
        return np.clip(predictions, 0.01, 0.99)

    def get_model_metrics(self, history_df):
        """Отримання метрик якості моделей"""
        X = history_df['year'].values.reshape(-1, 1)
        X_scaled = self.scaler.transform(X)

        metrics = []
        for i, model in enumerate(self.models):
            y_true = np.array([p[i] for p in history_df['probs']])
            y_pred = model.predict(X_scaled)

            metrics.append({
                "threat_id": i + 1,
                "mse": mean_squared_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
                "mae": np.mean(np.abs(y_true - y_pred))
            })

        return pd.DataFrame(metrics)

    def plot_forecast(self, history_df, forecast_years=3, org_type="Organization"):
        """Візуалізація прогнозу"""
        fig = go.Figure()

        # Історичні дані
        for i in range(len(history_df['probs'].iloc[0])):
            hist_probs = [p[i] for p in history_df['probs']]
            fig.add_trace(go.Scatter(
                x=history_df['year'],
                y=hist_probs,
                mode='lines+markers',
                name=f'Загроза {i + 1} (історія)',
                line=dict(width=2)
            ))

        # Прогноз
        last_year = history_df['year'].max()
        forecast_x = list(range(last_year + 1, last_year + forecast_years + 1))

        for i in range(len(self.models)):
            forecast_y = []
            for year in forecast_x:
                pred = self.predict(year)[i]
                forecast_y.append(pred)

            fig.add_trace(go.Scatter(
                x=forecast_x,
                y=forecast_y,
                mode='lines+markers',
                name=f'Загроза {i + 1} (прогноз)',
                line=dict(dash='dash', width=2)
            ))

        fig.update_layout(
            title=f'Прогноз ймовірностей загроз для {org_type}',
            xaxis_title='Рік',
            yaxis_title='Ймовірність',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        return fig