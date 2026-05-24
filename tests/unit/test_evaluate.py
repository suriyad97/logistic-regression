"""
Unit tests for evaluate module (src/evaluate.py).
"""
import pytest
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestEvaluate:
    """Test cases for evaluation process."""
    
    @pytest.fixture
    def trained_model_and_preprocessor(self, sample_large_data, tmp_path):
        from data_processing import DataProcessor
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        processor = DataProcessor()
        csv_path = tmp_path / "test_train.csv"
        sample_large_data.to_csv(csv_path, index=False)
        X_train, _, y_train, _ = processor.process_data(str(csv_path), is_training=True, test_size=0.4)
        
        model = LogisticRegression(max_iter=100)
        model.fit(X_train, y_train)
        
        model_path = tmp_path / "model.pkl"
        preproc_path = tmp_path / "preprocessor.joblib"
        
        joblib.dump(model, model_path)
        processor.save_preprocessor(str(preproc_path))
        
        return str(model_path), str(preproc_path), str(csv_path)

    def test_evaluate_model(self, trained_model_and_preprocessor, tmp_path):
        from evaluate import evaluate_model
        model_path, preproc_path, csv_path = trained_model_and_preprocessor
        
        output_dir = tmp_path / "eval_out"
        
        metrics = evaluate_model(model_path, preproc_path, csv_path, str(output_dir))
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics
        
        assert (output_dir / "evaluation_metrics.json").exists()
        assert (output_dir / "roc_curve.png").exists()
        assert (output_dir / "confusion_matrix.png").exists()

    def test_evaluate_model_no_seaborn(self, trained_model_and_preprocessor, tmp_path, monkeypatch):
        from evaluate import evaluate_model
        import evaluate
        
        model_path, preproc_path, csv_path = trained_model_and_preprocessor
        output_dir = tmp_path / "eval_out_no_sns"
        
        # Mock builtins.__import__ to fail on seaborn
        orig_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == 'seaborn':
                raise ImportError("Mocked seaborn import error")
            return orig_import(name, *args, **kwargs)
            
        monkeypatch.setattr('builtins.__import__', mock_import)
        
        # Also mock matplotlib to avoid Tkinter error in headless environment
        import matplotlib.pyplot as plt
        class MockAx:
            def matshow(self, *a, **k): return None
            def set_xticks(self, *a): pass
            def set_yticks(self, *a): pass
            def set_xticklabels(self, *a): pass
            def set_yticklabels(self, *a): pass
            def set_xlabel(self, *a): pass
            def set_ylabel(self, *a): pass
            def set_title(self, *a): pass
            def text(self, *a, **k): pass
            
        monkeypatch.setattr(plt, 'subplots', lambda *a, **k: (None, MockAx()))
        monkeypatch.setattr(plt, 'colorbar', lambda *a: None)
        def mock_savefig(path, *a, **k):
            with open(path, 'w') as f: f.write('fake png')
        monkeypatch.setattr(plt, 'savefig', mock_savefig)
        monkeypatch.setattr(plt, 'close', lambda *a: None)
        
        metrics = evaluate_model(model_path, preproc_path, csv_path, str(output_dir))
        assert (output_dir / "confusion_matrix.png").exists()

    def test_evaluate_model_exception(self, monkeypatch):
        from evaluate import evaluate_model
        
        with pytest.raises(Exception):
            evaluate_model("fake_model", "fake_preproc", "fake_data", "fake_out")

    def test_main(self, sample_large_data, tmp_path, monkeypatch):
        import sys
        import evaluate
        
        called = False
        def mock_evaluate_model(*args, **kwargs):
            nonlocal called
            called = True
            
        monkeypatch.setattr(evaluate, 'evaluate_model', mock_evaluate_model)
        
        data_path = tmp_path / "data.csv"
        sample_large_data.to_csv(data_path, index=False)
        
        test_args = ["evaluate.py", "--model", "model.pkl", "--preprocessor", "preprocessor.joblib", "--data", str(data_path), "--output", "outdir"]
        monkeypatch.setattr(sys, 'argv', test_args)
        
        # Exec logic
        evaluate.main()
        
        assert called is True
