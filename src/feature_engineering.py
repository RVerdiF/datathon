"""
Feature Engineering module for Passos Mágicos student data.
Creates additional features to improve model performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger("passos_magicos")


class FeatureEngineer:
    """Feature engineering for student risk prediction."""
    
    def __init__(self):
        self.created_features: List[str] = []
    
    def create_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate performance features."""
        df = df.copy()
        
        # Average of main indicators
        indicator_cols = ['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPV']
        existing_cols = [col for col in indicator_cols if col in df.columns]
        
        if existing_cols:
            df['avg_indicators'] = df[existing_cols].mean(axis=1)
            df['std_indicators'] = df[existing_cols].std(axis=1)
            df['min_indicators'] = df[existing_cols].min(axis=1)
            df['max_indicators'] = df[existing_cols].max(axis=1)
            self.created_features.extend(['avg_indicators', 'std_indicators', 'min_indicators', 'max_indicators'])
        
        # Average of grades
        grade_cols = ['Matem', 'Portug', 'Inglês']
        existing_grades = [col for col in grade_cols if col in df.columns]
        
        if existing_grades:
            df['avg_grades'] = df[existing_grades].mean(axis=1)
            self.created_features.append('avg_grades')
        
        return df
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to student engagement."""
        df = df.copy()
        
        # Number of evaluations as engagement proxy
        if 'Nº Av' in df.columns:
            df['high_engagement'] = (df['Nº Av'] >= df['Nº Av'].median()).astype(int)
            self.created_features.append('high_engagement')
        
        # Bolsa indication as recognition
        if 'Indicado' in df.columns:
            # Already encoded, keep as is
            pass
        
        return df
    
    def create_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on student trajectory over years."""
        df = df.copy()
        
        # Pedra evolution (if all years available)
        pedra_mapping = {'Quartzo': 1, 'Ágata': 2, 'Ametista': 3, 'Topázio': 4}
        
        for year in ['20', '21', '22']:
            col = f'Pedra {year}'
            if col in df.columns and df[col].dtype == 'object':
                df[f'pedra_num_{year}'] = df[col].map(pedra_mapping).fillna(0)
                self.created_features.append(f'pedra_num_{year}')
        
        # Pedra improvement (2022 vs 2021)
        if 'pedra_num_22' in df.columns and 'pedra_num_21' in df.columns:
            df['pedra_improvement'] = df['pedra_num_22'] - df['pedra_num_21']
            self.created_features.append('pedra_improvement')
        
        # Years in program
        if 'Ano ingresso' in df.columns:
            current_year = 2022
            df['years_in_program'] = current_year - df['Ano ingresso']
            self.created_features.append('years_in_program')
        
        return df
    
    def create_phase_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to student phase."""
        df = df.copy()
        
        # Phase vs ideal phase gap
        if 'Fase' in df.columns and 'Fase ideal' in df.columns:
            # If Fase ideal is categorical, try to extract number
            if df['Fase ideal'].dtype == 'object':
                df['fase_ideal_num'] = df['Fase ideal'].str.extract(r'(\d+)').astype(float)
            else:
                df['fase_ideal_num'] = df['Fase ideal']
            
            df['phase_gap'] = df['Fase'] - df['fase_ideal_num'].fillna(df['Fase'])
            self.created_features.extend(['fase_ideal_num', 'phase_gap'])
        
        # Advanced phase indicator
        if 'Fase' in df.columns:
            df['advanced_phase'] = (df['Fase'] >= 4).astype(int)
            self.created_features.append('advanced_phase')
        
        return df
    
    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk indicators."""
        df = df.copy()
        
        # Low performance indicator
        if 'INDE 22' in df.columns:
            inde_threshold = df['INDE 22'].quantile(0.25)
            df['low_inde'] = (df['INDE 22'] < inde_threshold).astype(int)
            self.created_features.append('low_inde')
        
        # Combination of risk factors
        risk_factors = []
        
        if 'IAN' in df.columns:
            risk_factors.append((df['IAN'] < df['IAN'].median()).astype(int))
        if 'IDA' in df.columns:
            risk_factors.append((df['IDA'] < df['IDA'].median()).astype(int))
        if 'IPS' in df.columns:
            risk_factors.append((df['IPS'] < df['IPS'].median()).astype(int))
        
        if risk_factors:
            df['risk_factor_count'] = sum(risk_factors)
            self.created_features.append('risk_factor_count')
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        self.created_features = []
        
        df = self.create_performance_features(df)
        df = self.create_engagement_features(df)
        df = self.create_trajectory_features(df)
        df = self.create_phase_features(df)
        df = self.create_risk_indicators(df)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of created feature names."""
        return self.created_features.copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering function.
    Takes a DataFrame and returns it with additional engineered features.
    """
    engineer = FeatureEngineer()
    df_engineered = engineer.create_all_features(df)
    
    logger.info(f"Created {len(engineer.created_features)} new features:")
    for feat in engineer.created_features:
        logger.info(f"  - {feat}")
    
    return df_engineered


if __name__ == '__main__':
    # Test feature engineering
    import pandas as pd
    
    df = pd.read_excel('data/raw/dados.xlsx')
    df_engineered = engineer_features(df)
    
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Engineered shape: {df_engineered.shape}")
