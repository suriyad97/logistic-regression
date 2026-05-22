"""
Data processing module for Titanic logistic regression model.
Handles data loading, cleaning, feature engineering, and preprocessing.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict
import joblib

# Pandera schema validation
try:
    from schema import validate_raw, validate_raw_inference, validate_processed
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False
    logger_init = logging.getLogger(__name__)
    logger_init.warning("Pandera not installed — schema validation skipped. pip install pandera")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, cleaning, and feature engineering."""
    
    def __init__(self, random_state: int = 42):
        """Initialize DataProcessor with random state for reproducibility."""
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names = None
        
    def load_data(self, data_path: str, is_training: bool = True) -> pd.DataFrame:
        """
        Load CSV data from file path and validate against schema.

        Args:
            data_path: Path to CSV file
            is_training: If True, validates against RawTitanicSchema (requires 'Survived').
                        If False, validates against InferenceRawTitanicSchema (Survived optional).

        Returns:
            Loaded DataFrame (validated)
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # ── Pandera raw schema validation ─────────────────────────
        if HAS_PANDERA:
            if is_training:
                result = validate_raw(df, raise_on_error=True)
                schema_name = "training"
            else:
                result = validate_raw_inference(df, raise_on_error=False)
                schema_name = "inference"
            
            if result.passed:
                logger.info(f"Raw data ({schema_name}) schema validation passed")
        # ─────────────────────────────────────────────────────────

        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        - Age: Fill with median
        - Embarked: Fill with mode
        - Cabin: Drop (too many missing)
        """
        logger.info("Handling missing values")
        df = df.copy()

        # Fill Age with median (CoW-safe assignment)
        df['Age'] = df['Age'].fillna(df['Age'].median())
        logger.info(f"Filled Age missing values with median: {df['Age'].median()}")

        # Fill Fare with median (Kaggle test.csv has 1 missing Fare)
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        logger.info(f"Filled Fare missing values with median: {df['Fare'].median()}")

        # Fill Embarked with mode
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        logger.info("Filled Embarked missing values with mode")

        # Drop Cabin — too many missing values (~77%)
        if 'Cabin' in df.columns:
            df = df.drop('Cabin', axis=1)

        return df
    
    def feature_engineering(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create and engineer features.
        
        Args:
            df: Input DataFrame
            is_training: If True, fit new encoders; if False, use existing
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Performing feature engineering")
        df = df.copy()
        
        # Extract title from Name — raw string fixes SyntaxWarning in Python 3.12+
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 
                                           'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                           'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Create family size feature
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Create IsAlone feature
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
        
        # Create AgeBand for binning age
        df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                labels=['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior'])
        
        # Create FareBand for binning fare using fixed bins (to support single-row inference)
        df['FareBand'] = pd.cut(df['Fare'], bins=[-0.001, 7.91, 14.45, 31.0, 1000.0], labels=['Q1', 'Q2', 'Q3', 'Q4'])

        # Fill any NaN produced by pd.cut / pd.qcut for edge values at inference.
        # Convert to string first so fillna works regardless of existing category list.
        df['AgeBand']  = df['AgeBand'].astype(str).replace('nan', 'Adult')
        df['FareBand'] = df['FareBand'].astype(str).replace('nan', 'Q2')

        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            df: Input DataFrame
            is_training: If True, fit new encoders; if False, use existing
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features")
        df = df.copy()

        categorical_features = ['Sex', 'Title', 'AgeBand', 'FareBand', 'Embarked']

        for feature in categorical_features:
            if is_training:
                self.label_encoders[feature] = LabelEncoder()
                df[feature] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                logger.info(f"Fitted LabelEncoder for {feature}: classes={list(self.label_encoders[feature].classes_)}")
            else:
                # At inference time, unseen labels (e.g. NaN becoming 'nan') are
                # mapped to the most frequent class seen during training so the
                # pipeline never crashes on real-world missing values.
                known_classes = set(self.label_encoders[feature].classes_)
                fallback = self.label_encoders[feature].classes_[0]  # lowest class = safest default
                col_str = df[feature].astype(str)
                col_safe = col_str.map(lambda v: v if v in known_classes else fallback)
                df[feature] = self.label_encoders[feature].transform(col_safe)
                logger.info(f"Applied existing LabelEncoder for {feature}")

        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for the model and validate the processed schema."""
        logger.info("Selecting features")

        features_to_keep = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                           'Title', 'FamilySize', 'IsAlone', 'AgeBand', 'FareBand', 'Embarked']

        df = df[features_to_keep]
        self.feature_names = df.columns.tolist()
        logger.info(f"Selected features: {self.feature_names}")

        # ── Pandera processed schema validation ───────────────────
        if HAS_PANDERA:
            # validate_processed is a soft warning (raise_on_error=False)
            # so a schema mismatch here flags a DataProcessor bug without
            # killing an otherwise valid training run
            validate_processed(df, raise_on_error=False)
        # ─────────────────────────────────────────────────────────

        return df
    
    def scale_features(self, X: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            X: Feature matrix
            is_training: If True, fit new scaler; if False, use existing
            
        Returns:
            Scaled feature matrix
        """
        logger.info("Scaling features")
        
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Fitted StandardScaler")
        else:
            X_scaled = self.scaler.transform(X)
            logger.info("Applied existing StandardScaler")
        
        return X_scaled
    
    def process_data(self, data_path: str, is_training: bool = True, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray, pd.Series]:
        """
        Complete data processing pipeline.
        
        Args:
            data_path: Path to CSV file
            is_training: If True, prepare train/test split; if False, prepare for inference
            test_size: Test set size fraction
            
        Returns:
            X_train, X_test, y_train, y_test, and feature names (if training)
        """
        # Load and clean data
        df = self.load_data(data_path, is_training=is_training)
        df = self.handle_missing_values(df)
        df = self.feature_engineering(df, is_training=is_training)
        df = self.encode_categorical_features(df, is_training=is_training)
        
        # Separate features and target
        y = df['Survived']
        df = self.select_features(df)
        
        # Scale features
        X = self.scale_features(df, is_training=is_training)
        
        if is_training:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            logger.info(f"Data split: Train {X_train.shape[0]}, Test {X_test.shape[0]}")
            
            return X_train, X_test, y_train, y_test
        else:
            # For inference, return all data
            logger.info(f"Processed {X.shape[0]} samples for inference")
            return X
    
    def save_preprocessor(self, save_path: str) -> None:
        """Save scaler and encoders for later use."""
        logger.info(f"Saving preprocessor to {save_path}")
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, save_path)
    
    def load_preprocessor(self, load_path: str) -> None:
        """Load previously saved scaler and encoders."""
        logger.info(f"Loading preprocessor from {load_path}")
        preprocessor = joblib.load(load_path)
        self.scaler = preprocessor['scaler']
        self.label_encoders = preprocessor['label_encoders']
        self.feature_names = preprocessor['feature_names']
