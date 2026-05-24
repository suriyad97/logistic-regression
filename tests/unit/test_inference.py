"""
Unit tests for inference module (src/inference.py).
"""
import pytest
import sys
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_processing import DataProcessor
from sklearn.linear_model import LogisticRegression


class TestDataProcessorInference:
    """Test cases for data processor inference workflow."""
    
    def test_processor_loading(self):
        """Test DataProcessor can be initialized for inference."""
        processor = DataProcessor()
        assert processor is not None
        assert hasattr(processor, 'load_data')
        assert hasattr(processor, 'handle_missing_values')
    
    def test_data_processing_pipeline(self, sample_large_data, tmp_path):
        """Test complete data processing pipeline."""
        # Save sample data to CSV
        csv_path = tmp_path / "test_inference.csv"
        sample_large_data.to_csv(csv_path, index=False)
        
        processor = DataProcessor()
        
        # Load and process
        df = processor.load_data(str(csv_path), is_training=False)
        df_cleaned = processor.handle_missing_values(df)
        
        assert df_cleaned.shape[0] == sample_large_data.shape[0]
        assert df_cleaned.isnull().sum().sum() == 0  # No missing values
    
    def test_batch_inference_preparation(self):
        """Test data preparation for batch inference."""
        # Create simple test data
        data = pd.DataFrame({
            "PassengerId": [1, 2, 3, 4, 5],
            "Pclass": [3, 1, 3, 1, 3],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05],
            "Embarked": ["S", "C", "S", "S", "Q"],
            "Survived": [0, 1, 1, 1, 0]
        })
        
        processor = DataProcessor()
        
        # Process data for inference
        df = data.copy()
        df_cleaned = processor.handle_missing_values(df)
        
        assert df_cleaned.shape[0] == len(data)
        assert df_cleaned.isnull().sum().sum() == 0


class TestModelPersistence:
    """Test cases for model loading and saving."""
    
    def test_model_saving_and_loading(self, tmp_path):
        """Test model can be saved and loaded."""
        # Train a simple model
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        
        model = LogisticRegression(max_iter=100)
        model.fit(X, y)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, str(model_path))
        
        # Load model
        loaded_model = joblib.load(str(model_path))
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
        # Verify loaded model produces same predictions
        pred_original = model.predict(X)
        pred_loaded = loaded_model.predict(X)
        
        assert (pred_original == pred_loaded).all()


class TestScoreValidation:
    """Test cases for score validation."""
    
    def test_prediction_output_range(self):
        """Test predictions are in valid range [0, 1]."""
        X = pd.DataFrame({
            "Pclass": [1, 2, 3, 1, 3],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05]
        })
        y = pd.Series([0, 1, 1, 1, 0])
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        probabilities = model.predict_proba(X)
        
        # Check all probabilities are in [0, 1]
        assert (probabilities >= 0).all() and (probabilities <= 1).all()
    
    def test_no_null_predictions(self):
        """Test no null values in predictions."""
        X = pd.DataFrame({
            "Pclass": [1, 2, 3, 1, 3],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05]
        })
        y = pd.Series([0, 1, 1, 1, 0])
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions is not None
        assert len(predictions) > 0
        assert not pd.isna(predictions).any()

class TestModelInferenceClass:
    """Test ModelInference class from inference.py."""
    
    @pytest.fixture
    def trained_model_and_preprocessor(self, sample_large_data, tmp_path):
        from data_processing import DataProcessor
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        processor = DataProcessor()
        
        # Actually better to just save preprocessor and mock model
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

    def test_predict_dataframe(self, trained_model_and_preprocessor, sample_large_data):
        from inference import ModelInference
        model_path, preproc_path, csv_path = trained_model_and_preprocessor
        
        inference = ModelInference(model_path, preproc_path)
        # Drop survived for inference
        inference_data = sample_large_data.drop('Survived', axis=1)
        results = inference.predict(inference_data)
        
        assert 'predictions' in results
        assert 'survival_probability' in results
        assert 'non_survival_probability' in results
        assert 'prediction_labels' in results
        assert len(results['predictions']) == len(inference_data)

    def test_predict_csv_path(self, trained_model_and_preprocessor):
        from inference import ModelInference
        model_path, preproc_path, csv_path = trained_model_and_preprocessor
        
        inference = ModelInference(model_path, preproc_path)
        results = inference.predict(csv_path)
        
        assert len(results['predictions']) > 0

    def test_predict_exception(self, trained_model_and_preprocessor, monkeypatch):
        from inference import ModelInference
        model_path, preproc_path, csv_path = trained_model_and_preprocessor
        
        inference = ModelInference(model_path, preproc_path)
        def mock_process(*args, **kwargs): raise ValueError("Test predict error")
        monkeypatch.setattr(inference.processor, 'load_data', mock_process)
        
        with pytest.raises(ValueError, match="Test predict error"):
            inference.predict(csv_path)

    def test_predict_single(self, trained_model_and_preprocessor, sample_large_data):
        from inference import ModelInference
        model_path, preproc_path, _ = trained_model_and_preprocessor
        
        inference = ModelInference(model_path, preproc_path)
        single_instance = sample_large_data.drop('Survived', axis=1).iloc[0].to_dict()
        result = inference.predict_single(single_instance)
        
        assert 'prediction_label' in result
        assert 'survival_probability' in result
        assert 'non_survival_probability' in result

    def test_predict_single_exception(self, trained_model_and_preprocessor, monkeypatch):
        from inference import ModelInference
        model_path, preproc_path, _ = trained_model_and_preprocessor
        
        inference = ModelInference(model_path, preproc_path)
        def mock_process(*args, **kwargs): raise ValueError("Test single predict error")
        monkeypatch.setattr(inference.processor, 'handle_missing_values', mock_process)
        
        with pytest.raises(ValueError, match="Test single predict error"):
            inference.predict_single({"Pclass": 1})

class TestBatchInferenceFunction:
    """Test batch_inference function."""
    
    def test_batch_inference_execution(self, tmp_path, sample_large_data):
        from inference import batch_inference
        from data_processing import DataProcessor
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        # Setup model, preprocessor, and data
        csv_path = tmp_path / "test_train.csv"
        sample_large_data.to_csv(csv_path, index=False)
        processor = DataProcessor()
        X_train, _, y_train, _ = processor.process_data(str(csv_path), is_training=True, test_size=0.4)
        
        model = LogisticRegression(max_iter=100)
        model.fit(X_train, y_train)
        
        model_path = tmp_path / "model.pkl"
        preproc_path = tmp_path / "preprocessor.joblib"
        output_path = tmp_path / "output.csv"
        
        joblib.dump(model, model_path)
        processor.save_preprocessor(str(preproc_path))
        
        # Run batch inference
        batch_inference(str(model_path), str(preproc_path), str(csv_path), str(output_path))
        
        # Validate output
        assert output_path.exists()
        out_df = pd.read_csv(output_path)
        assert 'prediction' in out_df.columns
        assert 'survival_probability' in out_df.columns
        assert 'prediction_label' in out_df.columns
        assert len(out_df) == len(sample_large_data)

    def test_batch_inference_exception(self, monkeypatch):
        from inference import batch_inference
        
        # Give invalid paths that will trigger an exception
        with pytest.raises(Exception):
            batch_inference("fake_model", "fake_preproc", "fake_in", "fake_out")

    def test_main_execution(self, sample_large_data, tmp_path, monkeypatch):
        # mock argparse execution
        import sys
        import inference
        
        data_path = tmp_path / "data.csv"
        sample_large_data.to_csv(data_path, index=False)
        
        test_args = ["inference.py", "--model", "model.pkl", "--preprocessor", "preprocessor.joblib", "--data", str(data_path), "--output", "outdir"]
        monkeypatch.setattr(sys, 'argv', test_args)
        
        # We need to mock batch_inference so it doesn't try to open dummy files
        import inference
        called = []
        def mock_batch(*args):
            called.append(args)
            
        monkeypatch.setattr(inference, 'batch_inference', mock_batch)
        
        # run the block
        inference.main()
        
        assert len(called) == 1
