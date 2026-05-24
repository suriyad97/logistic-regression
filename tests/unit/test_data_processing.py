"""
Unit tests for data processing module (src/data_processing.py).
"""
import pytest
import pandas as pd
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_processing import DataProcessor


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_data_processor_initialization(self):
        """Test DataProcessor can be initialized."""
        processor = DataProcessor(random_state=42)
        assert processor is not None
        assert processor.random_state == 42
        assert hasattr(processor, 'load_data')
        assert hasattr(processor, 'handle_missing_values')
    
    def test_load_data(self, sample_large_data, tmp_path):
        """Test data loading from CSV."""
        csv_path = tmp_path / "test_data.csv"
        sample_large_data.to_csv(csv_path, index=False)
        
        processor = DataProcessor()
        loaded_data = processor.load_data(str(csv_path), is_training=True)
        
        assert loaded_data.shape[0] > 0
        assert "Survived" in loaded_data.columns
        
        # also test inference load
        loaded_data_infer = processor.load_data(str(csv_path), is_training=False)
        assert loaded_data_infer.shape[0] > 0

    def test_load_data_exception(self):
        processor = DataProcessor()
        with pytest.raises(Exception):
            processor.load_data("nonexistent_file.csv")
            
    def test_pandera_missing(self, monkeypatch):
        import data_processing
        monkeypatch.setattr(data_processing, 'HAS_PANDERA', False)
        
        processor = data_processing.DataProcessor()
        df = pd.DataFrame({"A": [1]})
        monkeypatch.setattr(pd, 'read_csv', lambda x: df)
        
        result = processor.load_data("dummy.csv", is_training=True)
        assert result.shape == (1, 1)

    def test_handle_missing_values(self):
        """Test handling of missing values."""
        data = pd.DataFrame({
            "PassengerId": [1, 2, 3, 4, 5],
            "Pclass": [3, 1, 3, 1, 3],
            "Age": [22.0, None, 26.0, 35.0, 35.0],
            "Fare": [7.25, 71.2833, None, 53.1, 8.05],
            "Embarked": ["S", "C", None, "Q", "S"],
            "Survived": [0, 1, 1, 1, 0]
        })
        
        processor = DataProcessor()
        cleaned_data = processor.handle_missing_values(data)
        
        assert cleaned_data['Age'].isnull().sum() == 0
        assert cleaned_data['Fare'].isnull().sum() == 0
        
    def test_handle_missing_values_with_cabin(self):
        processor = DataProcessor()
        df = pd.DataFrame({
            'Age': [20, None, 30],
            'Fare': [10.0, None, 20.0],
            'Embarked': ['S', None, 'C'],
            'Cabin': ['C85', None, None]
        })
        result = processor.handle_missing_values(df)
        assert 'Cabin' not in result.columns

    def test_data_shape(self, sample_data):
        assert sample_data.shape[0] > 0
        assert sample_data.shape[1] > 0
    
    def test_missing_values_detection(self):
        data = pd.DataFrame({
            "feature1": [1, 2, None, 4],
            "feature2": [10, None, 30, 40]
        })
        missing_count = data.isnull().sum().sum()
        assert missing_count == 2
    
    def test_data_types(self, sample_data):
        assert sample_data["PassengerId"].dtype in ['int64', 'int32']
        assert sample_data["Age"].dtype == 'float64'

    def test_feature_engineering(self, sample_data):
        processor = DataProcessor()
        df = processor.feature_engineering(sample_data, is_training=True)
        assert 'Title' in df.columns
        assert 'FamilySize' in df.columns
        assert 'IsAlone' in df.columns
        assert 'AgeBand' in df.columns
        assert 'FareBand' in df.columns

    def test_encode_categorical_features(self, sample_data):
        processor = DataProcessor()
        df = processor.feature_engineering(sample_data, is_training=True)
        
        df_encoded = processor.encode_categorical_features(df, is_training=True)
        assert df_encoded['Sex'].dtype in ['int32', 'int64']
        assert 'Sex' in processor.label_encoders
        
        df_inference = df.copy()
        df_inference.loc[0, 'Sex'] = 'unseen_gender'
        df_inference_encoded = processor.encode_categorical_features(df_inference, is_training=False)
        assert df_inference_encoded['Sex'].dtype in ['int32', 'int64']
        
    def test_select_features(self, sample_data):
        processor = DataProcessor()
        df = processor.feature_engineering(sample_data, is_training=True)
        df = processor.encode_categorical_features(df, is_training=True)
        df_selected = processor.select_features(df)
        
        expected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                           'Title', 'FamilySize', 'IsAlone', 'AgeBand', 'FareBand', 'Embarked']
        assert list(df_selected.columns) == expected_features
        assert processor.feature_names == expected_features

    def test_scale_features(self, sample_data):
        processor = DataProcessor()
        df = processor.feature_engineering(sample_data, is_training=True)
        df = processor.encode_categorical_features(df, is_training=True)
        df = processor.select_features(df)
        
        scaled_train = processor.scale_features(df, is_training=True)
        assert scaled_train.shape == df.shape
        assert abs(scaled_train.mean()) < 0.1
        
        scaled_test = processor.scale_features(df, is_training=False)
        assert scaled_test.shape == df.shape

    def test_process_data_training(self, sample_large_data, tmp_path):
        csv_path = tmp_path / "test_train.csv"
        sample_large_data.to_csv(csv_path, index=False)
        
        processor = DataProcessor()
        X_train, X_test, y_train, y_test = processor.process_data(str(csv_path), is_training=True, test_size=0.4)
        
        assert X_train.shape[0] + X_test.shape[0] == len(sample_large_data)
        assert len(X_train) == len(y_train)

    def test_process_data_inference(self, sample_large_data, tmp_path):
        csv_path = tmp_path / "test_infer.csv"
        sample_large_data.to_csv(csv_path, index=False)
        
        processor = DataProcessor()
        processor.process_data(str(csv_path), is_training=True, test_size=0.4)
        
        X_infer = processor.process_data(str(csv_path), is_training=False)
        assert X_infer.shape[0] == len(sample_large_data)

    def test_save_load_preprocessor(self, sample_large_data, tmp_path):
        csv_path = tmp_path / "test_preproc.csv"
        sample_large_data.to_csv(csv_path, index=False)
        
        processor = DataProcessor()
        processor.process_data(str(csv_path), is_training=True)
        
        save_path = tmp_path / "preprocessor.joblib"
        processor.save_preprocessor(str(save_path))
        assert save_path.exists()
        
        new_processor = DataProcessor()
        new_processor.load_preprocessor(str(save_path))
        
        assert new_processor.feature_names == processor.feature_names
        assert list(new_processor.label_encoders.keys()) == list(processor.label_encoders.keys())
