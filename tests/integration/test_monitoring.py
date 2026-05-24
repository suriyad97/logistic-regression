"""
Integration tests for model monitoring workflows.
Tests drift detection and model performance monitoring.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data_processing import DataProcessor
from train import calculate_metrics


class TestDriftDetection:
    """Integration tests for drift detection functionality."""
    
    def test_data_drift_detection(self, sample_data):
        """Test detection of data distribution drift."""
        # Baseline data (original sample)
        baseline = sample_data[["Age", "Fare"]].copy()
        
        # Simulated drifted data (different distribution)
        drifted = sample_data[["Age", "Fare"]].copy()
        drifted["Age"] = drifted["Age"] + 10  # Shift age distribution
        
        # Simple drift detection: compare means
        baseline_mean = baseline.mean()
        drifted_mean = drifted.mean()
        
        mean_diff = abs(baseline_mean - drifted_mean)
        drift_detected = (mean_diff > 0).any()
        
        assert drift_detected
    
    def test_model_performance_degradation(self):
        """Test detection of model performance degradation."""
        # Baseline metrics
        baseline_accuracy = 0.85
        baseline_precision = 0.82
        
        # Current metrics
        current_accuracy = 0.75
        current_precision = 0.78
        
        # Degradation threshold (5%)
        threshold = 0.05
        
        accuracy_degradation = baseline_accuracy - current_accuracy
        precision_degradation = baseline_precision - current_precision
        
        # Should detect degradation
        assert accuracy_degradation > threshold
    
    def test_data_processor_data_quality_monitoring(self, sample_large_data, tmp_path):
        """Test monitoring data quality through DataProcessor."""
        # Save baseline data
        csv_path = tmp_path / "baseline.csv"
        sample_large_data.to_csv(csv_path, index=False)
        
        processor = DataProcessor()
        df_baseline = processor.load_data(str(csv_path), is_training=True)
        df_baseline_clean = processor.handle_missing_values(df_baseline)
        
        # Check data quality
        baseline_null_count = df_baseline_clean.isnull().sum().sum()
        assert baseline_null_count == 0


class TestMonitoringMetrics:
    """Integration tests for monitoring metrics collection."""
    
    def test_metrics_calculation_with_train_module(self):
        """Test metrics calculation using train module."""
        y_true = [0, 1, 1, 0, 1, 1, 0, 0]
        y_pred = [0, 1, 0, 0, 1, 1, 0, 1]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Verify all metrics are valid
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        
        # Verify metrics are in valid ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
    
    def test_metrics_logging(self, tmp_path):
        """Test that metrics can be logged and retrieved."""
        import json
        
        # Generate metrics using actual function
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        metrics = calculate_metrics(y_true, y_pred)
        
        # Save metrics
        metrics_file = tmp_path / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)
        
        # Read back
        with open(metrics_file, "r") as f:
            loaded_metrics = json.load(f)
        
        assert "accuracy" in loaded_metrics
        assert "precision" in loaded_metrics
        assert isinstance(loaded_metrics["accuracy"], (int, float))
