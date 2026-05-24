"""
Unit tests for score_batch module (src/score_batch.py).
"""
import pytest
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class TestScoreBatch:
    """Test cases for score_batch.py."""

    def test_init_missing_env(self, monkeypatch):
        import score_batch
        monkeypatch.delenv("AZUREML_MODEL_DIR", raising=False)
        with pytest.raises(ValueError, match="AZUREML_MODEL_DIR"):
            score_batch.init()

    def test_init_success_flat_structure(self, tmp_path, monkeypatch):
        import score_batch
        
        # Create dummy model files
        (tmp_path / "model.pkl").write_text("model")
        (tmp_path / "preprocessor.pkl").write_text("preproc")
        
        monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
        
        class MockInference:
            def __init__(self, m, p):
                self.m = m
                self.p = p
                
        monkeypatch.setattr(score_batch, 'ModelInference', MockInference)
        
        score_batch.init()
        
        assert score_batch.engine is not None
        assert score_batch.engine.m == str(tmp_path / "model.pkl")

    def test_init_success_nested_structure(self, tmp_path, monkeypatch):
        import score_batch
        
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.pkl").write_text("model")
        (model_dir / "preprocessor.pkl").write_text("preproc")
        
        monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
        
        class MockInference:
            def __init__(self, m, p):
                self.m = m
                
        monkeypatch.setattr(score_batch, 'ModelInference', MockInference)
        
        score_batch.init()
        assert score_batch.engine.m == str(model_dir / "model.pkl")

    def test_run_success(self, tmp_path, monkeypatch):
        import score_batch
        
        class MockEngine:
            def predict(self, df):
                return {
                    "predictions": [1, 0],
                    "survival_probability": [0.9, 0.1],
                    "prediction_labels": ["Survived", "Did not survive"]
                }
        
        score_batch.engine = MockEngine()
        
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"PassengerId": [100, 101], "Dummy": [1, 2]})
        df.to_csv(csv_path, index=False)
        
        results = score_batch.run([str(csv_path)])
        
        assert len(results) == 2
        assert "100,1,0.9000,Survived" in results[0]
        assert "101,0,0.1000,Did not survive" in results[1]

    def test_run_error(self, tmp_path, monkeypatch):
        import score_batch
        
        class MockEngine:
            def predict(self, df):
                raise ValueError("Dummy Error")
                
        score_batch.engine = MockEngine()
        
        csv_path = tmp_path / "data2.csv"
        pd.DataFrame({"PassengerId": [100]}).to_csv(csv_path, index=False)
        
        results = score_batch.run([str(csv_path)])
        
        assert len(results) == 1
        assert "ERROR:" in results[0]
        assert "Dummy Error" in results[0]
