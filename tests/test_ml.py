import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from core.ml_forecaster import ThreatForecaster, MODEL_CLASSES
from core.data_generator import CyberDataGenerator


class TestModelClasses:
    def test_all_models_instantiable(self):
        for name, cls in MODEL_CLASSES.items():
            model = cls() if callable(cls) and not isinstance(cls, type) else cls()
            assert model is not None, f"Failed to instantiate {name}"

    def test_model_classes_count(self):
        assert len(MODEL_CLASSES) == 8


class TestThreatForecaster:
    @pytest.fixture
    def forecaster(self):
        return ThreatForecaster(model_type="auto")

    @pytest.fixture
    def history_df(self):
        dg = CyberDataGenerator()
        return dg.generate_history("E-commerce", years=5)

    def test_train(self, forecaster, history_df):
        forecaster.train(history_df, n_threats=6)
        assert forecaster.is_fitted
        assert len(forecaster.models) == 6
        assert len(forecaster.best_model_names) == 6
        assert len(forecaster.residuals_std) == 6

    def test_predict(self, forecaster, history_df):
        forecaster.train(history_df, n_threats=6)
        preds = forecaster.predict(10)
        assert len(preds) == 6
        assert all(0.01 <= p <= 0.99 for p in preds)

    def test_monte_carlo(self, forecaster, history_df):
        forecaster.train(history_df, n_threats=6)
        sims = forecaster.predict_monte_carlo(10, n_simulations=100)
        assert sims.shape == (6, 100)
        assert np.all(sims >= 0.01)
        assert np.all(sims <= 0.99)

    def test_monte_carlo_correlated(self, forecaster, history_df):
        forecaster.train(history_df, n_threats=6)
        sims_corr = forecaster.predict_monte_carlo(10, n_simulations=100, correlated=True)
        sims_uncorr = forecaster.predict_monte_carlo(10, n_simulations=100, correlated=False)
        assert sims_corr.shape == sims_uncorr.shape

    def test_model_metrics(self, forecaster, history_df):
        forecaster.train(history_df, n_threats=6)
        metrics = forecaster.get_model_metrics(history_df)
        assert len(metrics) == 6
        assert "mse" in metrics.columns
        assert "r2" in metrics.columns
        assert "cv_r2" in metrics.columns

    def test_compare_all_models(self, forecaster, history_df):
        forecaster.train(history_df, n_threats=6)
        comp = forecaster.compare_all_models(history_df, n_threats=6)
        assert len(comp) > 0
        assert "Model" in comp.columns
        assert "R2" in comp.columns

    def test_feature_engineering(self, forecaster, history_df):
        forecaster.train(history_df, n_threats=6)
        assert forecaster.n_features > 1

    def test_correlation_matrix(self, forecaster, history_df):
        forecaster.train(history_df, n_threats=6)
        assert forecaster._correlation_matrix is not None
        assert forecaster._correlation_matrix.shape == (6, 6)

    def test_predict_before_train_raises(self):
        f = ThreatForecaster()
        with pytest.raises(ValueError):
            f.predict(10)


class TestLinearModel:
    def test_linear_model(self):
        dg = CyberDataGenerator()
        history = dg.generate_history("Bank", years=5)
        f = ThreatForecaster(model_type="linear")
        f.train(history, n_threats=6)
        preds = f.predict(10)
        assert len(preds) == 6
