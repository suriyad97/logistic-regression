"""
Unit tests for schema module (src/schema.py).
"""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from schema import validate_raw, validate_raw_inference, validate_processed, SchemaValidationResult, _parse_pandera_errors
import pandera as pa
import schema

class TestSchemaValidation:
    """Test cases for pandera schema validation."""

    def test_validate_raw_success(self, sample_large_data):
        result = validate_raw(sample_large_data, raise_on_error=True)
        assert result.passed is True
        assert result.schema_name == "RawTitanicSchema"

    def test_validate_raw_failure(self, sample_large_data):
        invalid_data = sample_large_data.drop('Survived', axis=1)
        result = validate_raw(invalid_data, raise_on_error=False)
        assert result.passed is False
        assert len(result.errors) > 0
        
        with pytest.raises(ValueError, match="RawTitanicSchema"):
            validate_raw(invalid_data, raise_on_error=True)

    def test_validate_raw_inference_success(self, sample_large_data):
        inference_data = sample_large_data.drop('Survived', axis=1)
        result = validate_raw_inference(inference_data, raise_on_error=True)
        assert result.passed is True
        assert result.schema_name == "InferenceRawTitanicSchema"

    def test_validate_raw_inference_failure(self, sample_large_data):
        invalid_data = sample_large_data.copy()
        invalid_data.loc[0, 'Pclass'] = 5
        result = validate_raw_inference(invalid_data, raise_on_error=False)
        assert result.passed is False
        assert len(result.errors) > 0
        
        # Hit raise_on_error = True
        with pytest.raises(ValueError):
            validate_raw_inference(invalid_data, raise_on_error=True)

    def test_validate_raw_inference_failure_no_failure_cases(self, sample_large_data, monkeypatch):
        # Hit the else branch (exc.failure_cases is None)
        orig_validate = schema.InferenceRawTitanicSchema.validate
        def mock_validate(*args, **kwargs):
            try:
                orig_validate(pd.DataFrame({'fake': [1]}), lazy=True)
            except pa.errors.SchemaErrors as e:
                e.failure_cases = None
                raise e
            
        monkeypatch.setattr(schema.InferenceRawTitanicSchema, 'validate', mock_validate)
        
        result = validate_raw_inference(sample_large_data, raise_on_error=False)
        assert result.passed is False

    def test_validate_processed_success(self):
        processed_data = pd.DataFrame({
            "Pclass": [1, 2, 3],
            "Sex": [0, 1, 0],
            "Age": [22.0, 38.0, 26.0],
            "SibSp": [1, 0, 0],
            "Parch": [0, 0, 0],
            "Fare": [7.25, 71.28, 7.92],
            "Title": [1, 2, 3],
            "FamilySize": [2, 1, 1],
            "IsAlone": [0, 1, 1],
            "AgeBand": [1, 2, 3],
            "FareBand": [1, 2, 3],
            "Embarked": [0, 1, 2]
        })
        result = validate_processed(processed_data, raise_on_error=True)
        assert result.passed is True

    def test_validate_processed_failure(self):
        invalid_data = pd.DataFrame({
            "Pclass": [1, 2, 3],
            "Sex": [0, 1, 0]
        })
        result = validate_processed(invalid_data, raise_on_error=False)
        assert result.passed is False
        
        with pytest.raises(ValueError):
            validate_processed(invalid_data, raise_on_error=True)

    def test_schema_validation_result(self):
        result = SchemaValidationResult(passed=False, schema_name="TestSchema", errors=[{"error": "test"}])
        assert result.to_dict()["passed"] is False
        assert result.to_dict()["schema_name"] == "TestSchema"
        assert result.to_dict()["error_count"] == 1
        
        with pytest.raises(ValueError):
            result.raise_if_failed()

    def test_parse_pandera_errors_exception(self):
        class MockSchemaError(Exception):
            pass
            
        errors = _parse_pandera_errors(MockSchemaError("Test Raw Error"))
        assert len(errors) == 1
        assert "raw_error" in errors[0]
        assert "Test Raw Error" in errors[0]["raw_error"]
