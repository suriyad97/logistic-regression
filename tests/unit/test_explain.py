"""
Unit tests for explain module (src/explain.py).
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class TestExplain:
    """Test cases for explain.py."""

    @pytest.fixture
    def setup_data(self, sample_large_data, tmp_path):
        from data_processing import DataProcessor
        from sklearn.linear_model import LogisticRegression
        
        processor = DataProcessor()
        csv_path = tmp_path / "test_train.csv"
        sample_large_data.to_csv(csv_path, index=False)
        X_train, X_test, y_train, _ = processor.process_data(str(csv_path), is_training=True, test_size=0.4)
        
        model = LogisticRegression(max_iter=100)
        model.fit(X_train, y_train)
        
        df_raw = processor.load_data(str(csv_path))
        df = processor.handle_missing_values(df_raw)
        df = processor.feature_engineering(df, is_training=False)
        df = processor.encode_categorical_features(df, is_training=False)
        X_test_df = df.iloc[:len(X_test)]
        
        return model, processor, X_train, X_test, X_test_df

    def test_explain_global_shap(self, setup_data, tmp_path, monkeypatch):
        from explain import ModelExplainer
        import explain
        
        model, processor, X_train, X_test, _ = setup_data
        explainer = ModelExplainer(model, processor, processor.feature_names)
        
        monkeypatch.setattr(explain, 'HAS_SHAP', False)
        result = explainer.explain_global_shap(X_train, X_test, tmp_path)
        assert result == {}
        
        class MockExplainer:
            def __init__(self, *args, **kwargs): pass
            # Return list to cover isinstance(shap_values, list)
            def shap_values(self, *args, **kwargs): 
                val = np.random.rand(len(X_test), len(processor.feature_names))
                return [val, val]
            
        class MockShap:
            LinearExplainer = MockExplainer
            def summary_plot(self, *args, **kwargs): pass
        
        class MockMLflow:
            def log_metric(self, *args, **kwargs): pass
            def log_artifact(self, *args, **kwargs): pass
            
        monkeypatch.setattr(explain, 'shap', MockShap(), raising=False)
        monkeypatch.setattr(explain, 'HAS_SHAP', True)
        monkeypatch.setattr(explain, 'HAS_MLFLOW', True)
        monkeypatch.setattr(explain, 'mlflow', MockMLflow(), raising=False)
        
        result = explainer.explain_global_shap(X_train, X_test, tmp_path)
        assert len(result) > 0

    def test_explain_local_shap(self, setup_data, tmp_path, monkeypatch):
        from explain import ModelExplainer
        import explain
        
        model, processor, X_train, X_test, X_test_df = setup_data
        explainer = ModelExplainer(model, processor, processor.feature_names)
        
        class MockExplainer:
            def __init__(self, *args, **kwargs): pass
            def shap_values(self, *args, **kwargs): 
                val = np.random.rand(len(X_test), len(processor.feature_names))
                return [val, val] # List return
            
        class MockShap:
            LinearExplainer = MockExplainer
        
        class MockMLflow:
            def log_artifact(self, *args, **kwargs): pass
            
        monkeypatch.setattr(explain, 'shap', MockShap(), raising=False)
        monkeypatch.setattr(explain, 'HAS_SHAP', True)
        monkeypatch.setattr(explain, 'HAS_MLFLOW', True)
        monkeypatch.setattr(explain, 'mlflow', MockMLflow(), raising=False)
        
        # Test indices=None
        explainer.explain_local_shap(X_train, X_test, X_test_df, tmp_path, indices=None)
        assert (tmp_path / "shap_waterfall_0.png").exists()

    def test_explain_local_lime(self, setup_data, tmp_path, monkeypatch):
        from explain import ModelExplainer
        import explain
        
        model, processor, X_train, X_test, X_test_df = setup_data
        explainer = ModelExplainer(model, processor, processor.feature_names)
        
        monkeypatch.setattr(explain, 'HAS_LIME', False)
        explainer.explain_local_lime(X_train, X_test, X_test_df, tmp_path, indices=[0])
        
        class MockExplanation:
            def save_to_file(self, *args, **kwargs): 
                with open(args[0], 'w') as f: f.write("mock")
            def as_pyplot_figure(self):
                import matplotlib.pyplot as plt
                return plt.figure()
                
        class MockLimeExplainer:
            def __init__(self, *args, **kwargs): pass
            def explain_instance(self, *args, **kwargs): return MockExplanation()
            
        class MockLimeTabular:
            LimeTabularExplainer = MockLimeExplainer
            
        class MockLime:
            lime_tabular = MockLimeTabular
            
        class MockMLflow:
            def log_artifact(self, *args, **kwargs): pass
            
        monkeypatch.setattr(explain, 'lime', MockLime(), raising=False)
        monkeypatch.setattr(explain, 'HAS_LIME', True)
        monkeypatch.setattr(explain, 'HAS_MLFLOW', True)
        monkeypatch.setattr(explain, 'mlflow', MockMLflow(), raising=False)
        
        # Test indices=None
        explainer.explain_local_lime(X_train, X_test, X_test_df, tmp_path, indices=None)
        assert (tmp_path / "lime_explanation_0.html").exists()

    def test_run_all_no_deps(self, setup_data, tmp_path, monkeypatch):
        from explain import ModelExplainer
        import explain
        
        model, processor, X_train, X_test, X_test_df = setup_data
        explainer = ModelExplainer(model, processor, processor.feature_names)
        
        monkeypatch.setattr(explain, 'HAS_SHAP', False)
        monkeypatch.setattr(explain, 'HAS_LIME', False)
        
        result = explainer.run_all(X_train, X_test, X_test_df, tmp_path, local_indices=[0])
        assert result == {}

    def test_main(self, setup_data, sample_large_data, monkeypatch):
        import sys
        import explain
        
        called = False
        class MockExplainer:
            def __init__(self, *args, **kwargs): pass
            def run_all(self, *args, **kwargs):
                nonlocal called
                called = True
                return {}
                
        monkeypatch.setattr(explain, 'ModelExplainer', MockExplainer)
        
        import joblib
        from pathlib import Path
        import tempfile
        
        model, processor, *_ = setup_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "m.pkl"
            proc_path = Path(tmpdir) / "p.joblib"
            data_path = Path(tmpdir) / "d.csv"
            
            joblib.dump(model, model_path)
            processor.save_preprocessor(str(proc_path))
            sample_large_data.to_csv(data_path, index=False)
            
            test_args = ["explain.py", "--model", str(model_path), "--preprocessor", str(proc_path), "--data", str(data_path), "--output", str(tmpdir)]
            monkeypatch.setattr(sys, 'argv', test_args)
            
            explain.main()
