"""
Unit tests for training module (src/train.py).
"""
import pytest
import sys
import pandas as pd
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import train
from train import calculate_metrics

class TestMetricsCalculation:
    def test_calculate_metrics_basic(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        metrics = calculate_metrics(y_true, y_pred)
        assert "accuracy" in metrics

    def test_metrics_with_probabilities(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        y_pred_proba = [0.2, 0.8, 0.3, 0.1, 0.9]
        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
        assert "roc_auc" in metrics

class TestTrainingProcess:

    def test_train_model_basic(self, sample_large_data, tmp_path, monkeypatch):
        data_path = tmp_path / "train.csv"
        sample_large_data.to_csv(data_path, index=False)
        output_dir = tmp_path / "model_out"
        
        class MockSklearn:
            @staticmethod
            def log_model(*args, **kwargs): pass
            
        class MockMLflow:
            sklearn = MockSklearn()
            def set_experiment(self, name): pass
            def start_run(self, **kwargs): 
                class Run:
                    info = type('Info', (), {'run_id': '123'})()
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                return Run()
            def end_run(self, **kwargs): pass
            def log_params(self, params): pass
            def log_metric(self, key, val): pass
            def log_artifacts(self, *args, **kwargs): pass
            def log_artifact(self, *args, **kwargs): pass
        
        monkeypatch.setattr(train, 'mlflow', MockMLflow())
        monkeypatch.setattr(train, 'HAS_MLFLOW', True)
        
        model, processor, metrics = train.train_model(
            data_path=str(data_path),
            output_dir=str(output_dir),
            test_size=0.4,
            tune=False,
            explain=True  # Cover explain=True branch
        )
        assert (output_dir / "model.pkl").exists()

    def test_train_model_tuning(self, sample_large_data, tmp_path, monkeypatch):
        data_path = tmp_path / "train.csv"
        sample_large_data.to_csv(data_path, index=False)
        output_dir = tmp_path / "model_out_tune"
        
        monkeypatch.setattr(train, 'HAS_MLFLOW', False)
        
        model, processor, metrics = train.train_model(
            data_path=str(data_path),
            output_dir=str(output_dir),
            test_size=0.4,
            tune=True,
            n_trials=2,
            explain=False
        )
        assert metrics["tuning"]["enabled"] is True
        assert (output_dir / "optuna_study.json").exists()

    def test_tune_hyperparameters_no_optuna(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train, 'HAS_OPTUNA', False)
        params = train.tune_hyperparameters(None, None, None, None)
        assert params["C"] == 1.0
        
        # Also cover _save_study_summary with HAS_OPTUNA = False
        train._save_study_summary(None, str(tmp_path))

    def test_make_objective(self, monkeypatch):
        X_train = np.random.rand(10, 5)
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        X_val = np.random.rand(4, 5)
        y_val = np.array([0, 1, 0, 1])
        
        # mock mlflow so we can cover the inner mlflow.start_run
        class MockMLflow:
            def start_run(self, **kwargs): 
                class Run:
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                return Run()
            def log_params(self, params): pass
            def log_metric(self, key, val): pass
            
        monkeypatch.setattr(train, 'mlflow', MockMLflow(), raising=False)
        monkeypatch.setattr(train, 'HAS_MLFLOW', True)
        
        objective = train._make_objective(X_train, y_train, X_val, y_val, "dummy_parent")
        
        class MockTrial:
            def __init__(self):
                self.number = 1
                self.user_attrs = {}
            def suggest_categorical(self, name, choices): return choices[0]
            def suggest_float(self, name, *args, **kwargs): return 1.0
            def suggest_int(self, name, *args, **kwargs): return 100
            def set_user_attr(self, name, val): self.user_attrs[name] = val
                
        trial = MockTrial()
        score = objective(trial)
        assert isinstance(score, float)
        assert "val_roc_auc" in trial.user_attrs

    def test_main(self, sample_large_data, tmp_path, monkeypatch):
        called = False
        def mock_train_model(*args, **kwargs):
            nonlocal called
            called = True
            
        monkeypatch.setattr(train, 'train_model', mock_train_model)
        
        data_path = tmp_path / "train.csv"
        sample_large_data.to_csv(data_path, index=False)
        
        test_args = ["train.py", "--data", str(data_path), "--output", "outdir", "--no-explain", "--tune"]
        monkeypatch.setattr(sys, 'argv', test_args)
        
        train.main()
        assert called is True

    def test_train_model_exception(self, tmp_path, monkeypatch):
        # Force an exception
        def mock_process(*args, **kwargs):
            raise ValueError("Test Exception")
            
        monkeypatch.setattr('train.DataProcessor.process_data', mock_process)
        
        class MockMLflow:
            def set_experiment(self, name): pass
            def start_run(self, **kwargs): 
                class Run:
                    info = type('Info', (), {'run_id': '123'})()
                return Run()
            def end_run(self, **kwargs): pass
            
        monkeypatch.setattr(train, 'mlflow', MockMLflow(), raising=False)
        monkeypatch.setattr(train, 'HAS_MLFLOW', True)
        
        with pytest.raises(ValueError, match="Test Exception"):
            train.train_model("fake.csv", str(tmp_path))

    def test_import_errors(self, monkeypatch):
        orig_import = __import__
        def mock_import(name, *args, **kwargs):
            if name in ('mlflow', 'mlflow.sklearn'): raise ImportError()
            if name in ('optuna', 'optuna.samplers'): raise ImportError()
            return orig_import(name, *args, **kwargs)
        monkeypatch.setattr('builtins.__import__', mock_import)
        
        # Reload train to hit top-level ImportError handlers
        import sys
        if 'train' in sys.modules:
            del sys.modules['train']
        import train
        assert train.HAS_MLFLOW is False
        assert train.HAS_OPTUNA is False

    def test_mlflow_logging_fallback(self, monkeypatch):
        import train
        monkeypatch.setattr(train, 'HAS_MLFLOW', False)
        def mock_process(*args, **kwargs):
            return np.random.rand(10,5), np.random.rand(4,5), np.array([0,1]*5), np.array([0,1]*2)
        monkeypatch.setattr('train.DataProcessor.process_data', mock_process)
        def mock_save(*args, **kwargs): pass
        monkeypatch.setattr('train.DataProcessor.save_preprocessor', mock_save)
        
        train.train_model("fake.csv", "fake_out", tune=False)

    def test_optuna_loop(self, monkeypatch):
        import train
        monkeypatch.setattr(train, 'HAS_OPTUNA', True)
        params = train.tune_hyperparameters(
            np.random.rand(10, 5), np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.random.rand(4, 5), np.array([0, 1, 0, 1]),
            n_trials=1, random_state=42
        )
        assert "C" in params

