import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
import plotly.graph_objects as go
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*R^2 score.*")

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "linear": LinearRegression,
    "ridge": lambda: Ridge(alpha=1.0),
    "lasso": lambda: Lasso(alpha=0.01),
    "elastic_net": lambda: ElasticNet(alpha=0.01),
    "random_forest": lambda: RandomForestRegressor(n_estimators=50, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=50, random_state=42),
    "svr": lambda: SVR(kernel="rbf", C=1.0, epsilon=0.01),
    "decision_tree": lambda: DecisionTreeRegressor(random_state=42),
}

MODEL_CLASSES_NAMED = {
    "Linear": LinearRegression,
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=0.01),
    "ElasticNet": lambda: ElasticNet(alpha=0.01),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=50, random_state=42),
    "SVR": lambda: SVR(kernel="rbf", C=1.0, epsilon=0.01),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=42),
}


class ThreatForecaster:
    def __init__(self, model_type="auto"):
        self.model_type = model_type
        self.models = []
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.is_fitted = False
        self.best_model_names = []
        self.residuals_std = []
        self.cv_scores = []
        self.n_features = 1
        self._correlation_matrix = None

    def _build_features(self, history_df, include_poly=True):
        years = history_df["year"].values.reshape(-1, 1)
        n_years = len(years)

        features = [years]

        if n_years >= 3:
            probs_matrix = np.array([p for p in history_df["probs"]])
            lag1 = np.roll(probs_matrix, 1, axis=0)
            lag1[0] = np.mean(probs_matrix, axis=0)
            features.append(lag1)

            rolling_mean = np.array([
                np.mean(probs_matrix[max(0, i - 2):i + 1], axis=0) for i in range(n_years)
            ])
            features.append(rolling_mean)

            if n_years >= 4:
                rolling_std = np.array([
                    np.std(probs_matrix[max(0, i - 2):i + 1], axis=0) for i in range(n_years)
                ])
                features.append(rolling_std)

        X = np.hstack(features)

        if include_poly and X.shape[1] <= 5:
            try:
                X_poly = self.poly.fit_transform(X[:, :1])
                X = np.hstack([X, X_poly[:, 1:]])
            except Exception:
                pass

        return X

    def _build_features_predict(self, year_value, last_probs=None):
        year_arr = np.array([[year_value]])
        features = [year_arr]

        if last_probs is not None:
            lag1 = np.array([last_probs])
            features.append(lag1)

            mean_probs = np.array([last_probs])
            features.append(mean_probs)

            if len(features) >= 3:
                std_probs = np.zeros_like(mean_probs)
                features.append(std_probs)

        X = np.hstack(features)

        if self.n_features > X.shape[1]:
            pad = np.zeros((1, self.n_features - X.shape[1]))
            X = np.hstack([X, pad])
        elif self.n_features < X.shape[1]:
            X = X[:, :self.n_features]

        return X

    def train(self, history_df, n_threats):
        self.models = []
        self.best_model_names = []
        self.residuals_std = []
        self.cv_scores = []

        X = self._build_features(history_df)
        self.n_features = X.shape[1]
        X_scaled = self.scaler.fit_transform(X)

        kf = KFold(n_splits=min(3, len(X)), shuffle=True, random_state=42) if len(X) >= 3 else None

        for i in range(n_threats):
            y = np.array([p[i] for p in history_df["probs"]])

            if self.model_type == "auto":
                best_model = None
                best_r2 = -float("inf")
                best_name = ""
                best_cv = 0

                for name, model_cls in MODEL_CLASSES.items():
                    try:
                        model = model_cls()
                        model.fit(X_scaled, y)
                        y_pred = model.predict(X_scaled)
                        r2 = r2_score(y, y_pred) if len(y) >= 2 else 0.0

                        cv_mean = 0
                        if kf and len(X) >= 3:
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring="r2")
                                    cv_mean = np.mean(cv_scores)
                            except Exception:
                                cv_mean = r2

                        score = cv_mean if kf else r2
                        if score > best_r2:
                            best_r2 = score
                            best_model = model
                            best_name = name
                            best_cv = cv_mean
                    except Exception as e:
                        logger.debug("Model %s failed for threat %d: %s", name, i + 1, e)
                        continue

                if best_model is None:
                    best_model = LinearRegression()
                    best_model.fit(X_scaled, y)
                    best_name = "linear"
                    best_cv = 0

                self.models.append(best_model)
                self.best_model_names.append(best_name)
                self.cv_scores.append(best_cv)
            else:
                model_cls = MODEL_CLASSES.get(self.model_type, LinearRegression)
                model = model_cls() if callable(model_cls) and not isinstance(model_cls, type) else model_cls()
                model.fit(X_scaled, y)
                self.models.append(model)
                self.best_model_names.append(self.model_type)
                self.cv_scores.append(0)

            residuals = y - self.models[-1].predict(X_scaled)
            self.residuals_std.append(np.std(residuals) if np.std(residuals) > 0 else 0.01)

        self._compute_correlation(history_df)
        self.is_fitted = True
        return self

    def _compute_correlation(self, history_df):
        probs_matrix = np.array([p for p in history_df["probs"]])
        if probs_matrix.shape[0] >= 3:
            self._correlation_matrix = np.corrcoef(probs_matrix.T)
            np.nan_to_num(self._correlation_matrix, copy=False, nan=0.0)
        else:
            n = probs_matrix.shape[1]
            self._correlation_matrix = np.eye(n)

    def predict(self, year_value, last_probs=None):
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_pred = self._build_features_predict(year_value, last_probs)
        if X_pred.shape[1] < self.n_features:
            pad = np.zeros((1, self.n_features - X_pred.shape[1]))
            X_pred = np.hstack([X_pred, pad])
        elif X_pred.shape[1] > self.n_features:
            X_pred = X_pred[:, :self.n_features]

        X_scaled = self.scaler.transform(X_pred)
        predictions = [model.predict(X_scaled)[0] for model in self.models]
        return np.clip(predictions, 0.01, 0.99)

    def predict_monte_carlo(self, year_value, n_simulations=1000, correlated=True, last_probs=None):
        base_preds = self.predict(year_value, last_probs)
        n_threats = len(self.models)

        if correlated and self._correlation_matrix is not None and n_threats > 1:
            try:
                L = np.linalg.cholesky(self._correlation_matrix)
                independent_samples = np.random.standard_normal((n_threats, n_simulations))
                correlated_samples = L @ independent_samples
            except np.linalg.LinAlgError:
                correlated_samples = np.random.standard_normal((n_threats, n_simulations))
        else:
            correlated_samples = np.random.standard_normal((n_threats, n_simulations))

        simulations = np.zeros((n_threats, n_simulations))
        for i in range(n_threats):
            sims = base_preds[i] + self.residuals_std[i] * correlated_samples[i]
            sims = np.clip(sims, 0.01, 0.99)
            simulations[i] = sims

        return simulations

    def get_model_metrics(self, history_df):
        X = self._build_features(history_df)
        X_scaled = self.scaler.transform(X)
        metrics = []
        for i, model in enumerate(self.models):
            y_true = np.array([p[i] for p in history_df["probs"]])
            y_pred = model.predict(X_scaled)
            metrics.append({
                "threat_id": i + 1,
                "model": self.best_model_names[i],
                "mse": mean_squared_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred) if len(y_true) >= 2 else 0.0,
                "mae": mean_absolute_error(y_true, y_pred),
                "cv_r2": self.cv_scores[i] if i < len(self.cv_scores) else 0,
            })
        return pd.DataFrame(metrics)

    def compare_all_models(self, history_df, n_threats):
        X = self._build_features(history_df)
        X_scaled = StandardScaler().fit_transform(X)
        results = []
        for i in range(n_threats):
            y = np.array([p[i] for p in history_df["probs"]])
            for name, model_cls in MODEL_CLASSES_NAMED.items():
                try:
                    model = model_cls() if callable(model_cls) and not isinstance(model_cls, type) else model_cls()
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    results.append({
                        "Threat ID": i + 1,
                        "Model": name,
                        "MSE": mean_squared_error(y, y_pred),
                        "R2": r2_score(y, y_pred) if len(y) >= 2 else 0.0,
                    })
                except Exception:
                    continue
        return pd.DataFrame(results)

    def plot_forecast(self, history_df, forecast_years=3, org_type="Organization", n_simulations=1000):
        fig = go.Figure()
        last_year = history_df["year"].max()
        forecast_x = list(range(last_year + 1, last_year + forecast_years + 1))

        n_threats = len(history_df["probs"].iloc[0])
        last_probs = history_df["probs"].iloc[-1] if len(history_df) > 0 else None

        for i in range(n_threats):
            hist_probs = [p[i] for p in history_df["probs"]]
            fig.add_trace(go.Scatter(
                x=history_df["year"], y=hist_probs, mode="lines+markers",
                name=f"Загроза {i + 1}", line=dict(width=2),
            ))

            forecast_y = []
            lower_bound = []
            upper_bound = []
            for year in forecast_x:
                sims = self.predict_monte_carlo(year, n_simulations, last_probs=last_probs)[i]
                forecast_y.append(np.mean(sims))
                lower_bound.append(np.percentile(sims, 5))
                upper_bound.append(np.percentile(sims, 95))

            fig.add_trace(go.Scatter(
                x=forecast_x, y=forecast_y, mode="lines+markers",
                name=f"Прогноз {i + 1} ({self.best_model_names[i]})",
                line=dict(dash="dash", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=forecast_x + forecast_x[::-1],
                y=upper_bound + lower_bound[::-1],
                fill="toself", fillcolor="rgba(0,100,80,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"ДІ 90% {i + 1}", showlegend=False,
            ))

        fig.update_layout(
            title=f"Прогноз для {org_type} (Monte Carlo, N={n_simulations})",
            xaxis_title="Рік", yaxis_title="Ймовірність",
            hovermode="x unified", template="plotly_white", height=600,
        )
        return fig
