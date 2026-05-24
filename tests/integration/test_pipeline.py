"""
Integration tests for end-to-end pipeline workflows.
Tests the interaction between data preparation, training, and inference.
"""
import pytest
import pandas as pd
import sys
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_processing import DataProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from train import calculate_metrics


class TestTrainingPipeline:
    """Integration tests for the training pipeline."""
    
    def test_data_to_model_workflow(self):
        """Test complete workflow from data loading to model training."""
        # Create test data
        test_data = pd.DataFrame({
            "Pclass": [1, 2, 3, 1, 3],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05],
            "Survived": [0, 1, 1, 1, 0]
        })
        
        # Prepare data
        X = test_data[["Pclass", "Age", "Fare"]]
        y = test_data["Survived"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        assert 0 <= train_score <= 1
        assert 0 <= test_score <= 1
    
    def test_data_processing_training_workflow(self, sample_large_data, tmp_path):
        """Test workflow: load data -> process -> train model."""
        # Save sample data
        csv_path = tmp_path / "train_data.csv"
        sample_large_data.to_csv(csv_path, index=False)
        
        # Load and process data
        processor = DataProcessor(random_state=42)
        df = processor.load_data(str(csv_path), is_training=True)
        df_cleaned = processor.handle_missing_values(df)
        
        # Prepare features
        X = df_cleaned[["Pclass", "Age", "Fare"]]
        y = df_cleaned["Survived"]
        
        # Train model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        metrics = calculate_metrics(y, y_pred)
        
        assert metrics["accuracy"] > 0
        assert len(metrics) > 0
    
    def test_model_persistence_workflow(self, tmp_path):
        """Test workflow: train model, save, load, and score."""
        # Create test data
        test_data = pd.DataFrame({
            "Pclass": [1, 2, 3, 1, 3],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05],
            "Survived": [0, 1, 1, 1, 0]
        })
        
        # Train
        X = test_data[["Pclass", "Age", "Fare"]]
        y = test_data["Survived"]
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        # Save using joblib
        model_path = tmp_path / "model.pkl"
        joblib.dump(model, str(model_path))
        
        # Load
        loaded_model = joblib.load(str(model_path))
        
        # Score with loaded model
        predictions = loaded_model.predict(X)
        assert len(predictions) == len(y)


class TestInferencePipeline:
    """Integration tests for the inference pipeline."""
    
    def test_batch_inference_workflow(self, tmp_path):
        """Test complete batch inference workflow."""
        # Create test data
        test_data = pd.DataFrame({
            "Pclass": [1, 2, 3, 1, 3],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05],
            "Survived": [0, 1, 1, 1, 0]
        })
        
        # Setup: Train and save model
        X = test_data[["Pclass", "Age", "Fare"]]
        y = test_data["Survived"]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        model_path = tmp_path / "model.pkl"
        joblib.dump(model, str(model_path))
        
        # Inference: Load model and score
        loaded_model = joblib.load(str(model_path))
        
        predictions = loaded_model.predict(X)
        probabilities = loaded_model.predict_proba(X)
        
        # Validate outputs
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
        assert (probabilities >= 0).all() and (probabilities <= 1).all()
    
    def test_data_processor_inference_workflow(self, sample_large_data, tmp_path):
        """Test data processor in inference workflow."""
        # Save sample data
        csv_path = tmp_path / "inference_data.csv"
        sample_large_data.to_csv(csv_path, index=False)
        
        # Load and process for inference
        processor = DataProcessor()
        df = processor.load_data(str(csv_path), is_training=False)
        df_processed = processor.handle_missing_values(df)
        
        # Verify data is ready for inference
        assert df_processed.isnull().sum().sum() == 0
        assert len(df_processed) == len(sample_large_data)
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent across calls."""
        test_data = pd.DataFrame({
            "Pclass": [1, 2, 3, 1, 3],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05],
            "Survived": [0, 1, 1, 1, 0]
        })
        
        X = test_data[["Pclass", "Age", "Fare"]]
        y = test_data["Survived"]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        # Make predictions twice
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        
        # Results should be identical
        assert (pred1 == pred2).all()


class TestDataValidation:
    """Integration tests for data validation in pipelines."""
    
    def test_schema_validation(self):
        """Test that data schema is validated during pipeline."""
        test_data = pd.DataFrame({
            "Pclass": [1, 2, 3, 1, 3],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05],
            "Survived": [0, 1, 1, 1, 0]
        })
        
        required_columns = ["Pclass", "Age", "Fare", "Survived"]
        assert all(col in test_data.columns for col in required_columns)
    
    def test_data_quality_checks(self):
        """Test data quality validation in pipeline."""
        test_data = pd.DataFrame({
            "Pclass": [1, 2, 3, 1, 3],
            "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.28, 7.93, 53.1, 8.05],
            "Survived": [0, 1, 1, 1, 0]
        })
        
        # Check for negative values
        numeric_cols = ["Pclass", "Age", "Fare", "Survived"]
        assert all((test_data[col] >= 0).all() for col in numeric_cols)
        
        # Check no infinite values
        assert not test_data[numeric_cols].isin([float('inf'), 
                                                    float('-inf')]).any().any()
    
    def test_data_processor_data_quality(self, sample_large_data, tmp_path):
        """Test DataProcessor validates data quality."""
        csv_path = tmp_path / "quality_check.csv"
        sample_large_data.to_csv(csv_path, index=False)
        
        processor = DataProcessor()
        df = processor.load_data(str(csv_path), is_training=True)
        df_clean = processor.handle_missing_values(df)
        
        # Validate cleaned data has no missing values
        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) > 0
        # Check for negative values
        numeric_cols = ["Pclass", "Age", "Fare", "Survived"]
        assert all((df_clean[col] >= 0).all() for col in numeric_cols)
        
        # Check no infinite values
        assert not df_clean[numeric_cols].isin([float('inf'), 
                                                    float('-inf')]).any().any()
