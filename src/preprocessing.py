"""
Preprocessing module for Passos Mágicos student data.
Handles data loading, cleaning, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import joblib
from pathlib import Path


class DataPreprocessor:
    """Preprocessor for student risk prediction data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: list = []
        self.target_column = 'target'
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from Excel file."""
        df = pd.read_excel(filepath)
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable.
        Risco (1): Defas < 0 (student is behind)
        Sem Risco (0): Defas >= 0 (student is on track or ahead)
        """
        df = df.copy()
        df[self.target_column] = (df['Defas'] < 0).astype(int)
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns: fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('Unknown')
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        categorical_cols = [
            'Gênero', 'Instituição de ensino', 'Pedra 20', 'Pedra 21', 'Pedra 22',
            'Indicado', 'Atingiu PV', 'Fase ideal', 'Destaque IEG', 'Destaque IDA', 'Destaque IPV'
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    # Handle unknown categories
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str)
                        # Handle unseen categories
                        df[col] = df[col].apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for modeling."""
        # Columns to drop (identifiers, redundant, or target-related)
        drop_cols = [
            'RA', 'Nome', 'Turma', 'Defas',  # ID and target-related
            'Avaliador1', 'Avaliador2', 'Avaliador3', 'Avaliador4',  # Names
            'Rec Av1', 'Rec Av2', 'Rec Av3', 'Rec Av4', 'Rec Psicologia',  # Text recommendations
            'Cg', 'Cf', 'Ct',  # Rankings (derived from target)
        ]
        
        # Keep only columns that exist
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        df = df.drop(columns=drop_cols, errors='ignore')
        
        # Store feature columns (excluding target)
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features."""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps (fit mode)."""
        df = self.create_target(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df, fit=True)
        df = self.select_features(df)
        df = self.scale_features(df, fit=True)
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to new data (transform only)."""
        df = self.create_target(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df, fit=False)
        df = self.select_features(df)
        df = self.scale_features(df, fit=False)
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets."""
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save(self, filepath: str):
        """Save the preprocessor to disk."""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Load a preprocessor from disk."""
        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.scaler = data['scaler']
        preprocessor.label_encoders = data['label_encoders']
        preprocessor.feature_columns = data['feature_columns']
        preprocessor.target_column = data['target_column']
        return preprocessor


def preprocess_data(
    input_path: str,
    output_dir: str = 'data/processed',
    preprocessor_path: str = 'models/preprocessor.joblib'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing function.
    Loads raw data, applies preprocessing, and saves processed data.
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize and apply preprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(input_path)
    df_processed = preprocessor.fit_transform(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)
    
    # Create DataFrames with target
    train_df = X_train.copy()
    train_df['target'] = y_train
    
    val_df = X_val.copy()
    val_df['target'] = y_val
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    # Save processed data
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    # Save preprocessor
    preprocessor.save(preprocessor_path)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Features: {len(preprocessor.feature_columns)}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    train_df, val_df, test_df = preprocess_data('data/raw/dados.xlsx')
