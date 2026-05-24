"""
Pytest configuration and shared fixtures for unit and integration tests.
"""
import pytest
import sys
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_data():
    """Fixture providing sample Titanic data for testing."""
    import pandas as pd
    df = pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Pclass": [3, 1, 3, 1, 3],
        "Name": ["Passenger 1", "Passenger 2", "Passenger 3", "Passenger 4", "Passenger 5"],
        "Sex": ["male", "female", "female", "female", "male"],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0],
        "SibSp": [1, 1, 0, 1, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", "373450"],
        "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
        "Embarked": ["S", "C", "S", "S", "S"],
        "Survived": [0, 1, 1, 1, 0]
    })
    return df


@pytest.fixture
def sample_model_output():
    """Fixture providing sample model predictions."""
    return [0.2, 0.8, 0.7, 0.9, 0.3]


@pytest.fixture
def sample_large_data():
    """Fixture providing large sample Titanic data (500+ rows) for schema validation."""
    import pandas as pd
    # Create large dataset with all required columns for schema validation
    n_rows = 550
    np.random.seed(42)
    data = {
        "PassengerId": np.arange(1, n_rows + 1),
        "Pclass": np.random.choice([1, 2, 3], n_rows),
        "Name": [f"Passenger {i+1}" for i in range(n_rows)],
        "Sex": np.random.choice(["male", "female"], n_rows),
        "Age": np.maximum(0, np.random.normal(30, 15, n_rows)),  # Ensure Age >= 0
        "SibSp": np.random.choice([0, 1, 2, 3], n_rows),
        "Parch": np.random.choice([0, 1, 2], n_rows),
        "Ticket": [f"TICKET{i+1}" for i in range(n_rows)],
        "Fare": np.maximum(0, np.random.normal(50, 30, n_rows)),  # Ensure Fare >= 0
        "Embarked": np.random.choice(["S", "C", "Q"], n_rows),
        "Survived": np.random.choice([0, 1], n_rows)
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Fixture providing a temporary directory for test data."""
    return tmp_path
