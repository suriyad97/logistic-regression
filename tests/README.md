# Testing Guide

This directory contains unit and integration tests for the MLOps project.

## Structure

```
tests/
├── conftest.py              # Shared pytest fixtures and configuration
├── unit/                    # Unit tests for individual modules
│   ├── test_data_processing.py  # Tests for data_processing.py
│   ├── test_train.py            # Tests for train.py
│   └── test_inference.py        # Tests for inference.py
└── integration/             # Integration tests for end-to-end workflows
    ├── test_pipeline.py     # Tests for training and inference pipelines
    └── test_monitoring.py   # Tests for monitoring and drift detection
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/unit/test_data_processing.py
```

### Run specific test class
```bash
pytest tests/unit/test_data_processing.py::TestDataProcessing
```

### Run specific test function
```bash
pytest tests/unit/test_data_processing.py::TestDataProcessing::test_load_data_success
```

### Run with markers
```bash
pytest -m unit
pytest -m integration
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

## Test Categories

### Unit Tests (`tests/unit/`)
Test individual functions and components in isolation:
- **test_data_processing.py**: Data loading, validation, feature engineering
- **test_train.py**: Model training, evaluation metrics
- **test_inference.py**: Model loading, scoring, validation

### Integration Tests (`tests/integration/`)
Test end-to-end workflows and component interactions:
- **test_pipeline.py**: Data → Training → Inference workflows
- **test_monitoring.py**: Drift detection, metrics logging, performance monitoring

## Fixtures

Common fixtures are defined in `conftest.py`:
- `sample_data`: Sample DataFrame with Titanic-like structure
- `sample_model_output`: Sample prediction outputs
- `temp_data_dir`: Temporary directory for test data

## Best Practices

1. **Keep tests focused**: Each test should verify one thing
2. **Use descriptive names**: Test names should explain what they test
3. **Use fixtures**: Leverage pytest fixtures for setup/teardown
4. **Mock external dependencies**: Use mocks for external services
5. **Test both happy and sad paths**: Include error cases
6. **Keep tests fast**: Unit tests should run quickly
7. **Make tests independent**: Tests shouldn't depend on execution order

## CI/CD Integration

These tests will be integrated into your Azure DevOps pipelines:
- Run during `ci-setup-environment.yml` for smoke tests
- Expand in `ct-train-register-model.yml` for validation before registration
- Add to quality gates in `qa-champion-challenger.yml`

## Next Steps

1. Implement actual test cases based on your modules' logic
2. Add more comprehensive edge cases
3. Add performance tests for slow operations
4. Set up code coverage targets (e.g., 80%+)
5. Integrate into CI/CD pipeline
