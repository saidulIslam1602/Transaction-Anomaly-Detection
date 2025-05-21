import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional


class TransactionPreprocessor:
    """
    Data preprocessing class for AML transaction monitoring.
    
    This class handles loading, cleaning, and feature engineering for
    transaction data used in AML monitoring and fraud detection.
    """
    
    def __init__(self, 
                 categorical_features: List[str] = None,
                 numerical_features: List[str] = None,
                 target: str = 'isFraud'):
        """
        Initialize the preprocessor with feature specifications.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            target: Name of the target variable
        """
        self.categorical_features = categorical_features or ['type']
        self.numerical_features = numerical_features or ['amount', 'oldbalanceOrg', 
                                                        'newbalanceOrig', 'oldbalanceDest', 
                                                        'newbalanceDest']
        self.target = target
        self.scaler = StandardScaler()
        self.encoders = {}
        
    def load_data(self, 
                  filepath: str, 
                  sample: Optional[int] = None) -> pd.DataFrame:
        """
        Load transaction data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            sample: Number of samples to load (for testing)
            
        Returns:
            DataFrame containing the loaded data
        """
        if sample:
            df = pd.read_csv(filepath, nrows=sample)
        else:
            df = pd.read_csv(filepath)
            
        print(f"Loaded data with {df.shape[0]} transactions and {df.shape[1]} features")
        return df
    
    def preprocess(self, 
                  df: pd.DataFrame, 
                  train: bool = True) -> pd.DataFrame:
        """
        Clean and preprocess the transaction data.
        
        Args:
            df: Raw transaction DataFrame
            train: Whether this is training data (vs inference data)
            
        Returns:
            Preprocessed DataFrame
        """
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Detect and handle outliers
        df = self._handle_outliers(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # For numerical features, replace missing values with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical features, replace missing values with most frequent
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
                
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones."""
        # Check if old balance equals new balance plus amount
        df['errorBalanceOrig'] = (df.newbalanceOrig != df.oldbalanceOrg - df.amount).astype(int)
        
        # Check for zero balances (suspicious in some contexts)
        df['isZeroBalanceOrig'] = (df.oldbalanceOrg == 0).astype(int)
        df['isZeroBalanceDest'] = (df.oldbalanceDest == 0).astype(int)
        
        # Transaction amount relative to balances
        df['amountToBalanceRatio'] = df.amount / df.oldbalanceOrg.replace(0, 0.01)
        
        # Flag for when destination account balance doesn't change despite receiving funds
        df['destBalanceUnchanged'] = ((df.newbalanceDest == df.oldbalanceDest) & 
                                      (df.amount > 0)).astype(int)
        
        # Transaction velocity features
        if 'step' in df.columns:
            entity_velocity = df.groupby(['nameOrig', 'step']).size().reset_index()
            entity_velocity.columns = ['nameOrig', 'step', 'txn_count']
            df = pd.merge(df, entity_velocity, on=['nameOrig', 'step'], how='left')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in numerical features."""
        # Use IQR method for outlier detection
        for col in self.numerical_features:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Cap extreme values rather than removing them
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Create outlier flag features
                df[f'{col}_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
                
                # Cap the values
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                
        return df
    
    def encode_and_scale(self, 
                         df: pd.DataFrame, 
                         train: bool = True) -> pd.DataFrame:
        """
        Encode categorical features and scale numerical features.
        
        Args:
            df: Preprocessed DataFrame
            train: Whether this is training data (to fit or just transform)
            
        Returns:
            DataFrame with encoded and scaled features
        """
        result_df = df.copy()
        
        # One-hot encode categorical features
        for col in self.categorical_features:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                result_df = pd.concat([result_df, dummies], axis=1)
                result_df.drop(col, axis=1, inplace=True)
        
        # Scale numerical features
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        if numerical_cols:
            if train:
                scaled_features = self.scaler.fit_transform(df[numerical_cols])
            else:
                scaled_features = self.scaler.transform(df[numerical_cols])
                
            scaled_df = pd.DataFrame(scaled_features, columns=numerical_cols, index=df.index)
            
            # Replace original columns with scaled versions
            for col in numerical_cols:
                result_df[col] = scaled_df[col]
        
        return result_df
    
    def prepare_train_test_data(self, 
                               df: pd.DataFrame, 
                               test_size: float = 0.2, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.
        
        Args:
            df: Preprocessed DataFrame
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in DataFrame")
            
        # Identify non-feature columns to exclude
        exclude_cols = ['step', 'nameOrig', 'nameDest', self.target]
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        
        # Split the data
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        y = df[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test 