import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

class DataPreprocessor:
    """Data preprocessing and validation for invoice data."""
    
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        self.preprocessor = None
        self.feature_names = None
    
    def identify_features(self, df, target_cols):
        """Identify numeric and categorical features."""
        exclude_cols = ['invoice_id', 'invoice_date'] + target_cols
        
        self.numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                 if col not in exclude_cols]
        self.categorical_features = [col for col in df.select_dtypes(include=['object']).columns 
                                     if col not in exclude_cols]
        
        print(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        return self.numeric_features, self.categorical_features
    
    def create_preprocessor(self):
        """Create sklearn preprocessing pipeline."""
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        
        return self.preprocessor
    
    def prepare_data(self, df, target_col, test_size=0.2, random_state=42):
        """Prepare features and target, split into train/test."""
        
        X = df.drop(columns=[target_col, 'invoice_id', 'invoice_date'], errors='ignore')
        y = df[target_col]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, X_train):
        """Fit preprocessor and transform training data."""
        if self.preprocessor is None:
            self.create_preprocessor()
        
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Get feature names after one-hot encoding
        cat_feature_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
            self.categorical_features
        )
        self.feature_names = list(self.numeric_features) + list(cat_feature_names)
        
        return X_train_processed
    
    def transform(self, X):
        """Transform data using fitted preprocessor."""
        return self.preprocessor.transform(X)
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Save preprocessor to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'preprocessor': self.preprocessor,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names
        }, path)
        print(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path='models/preprocessor.pkl'):
        """Load preprocessor from disk."""
        data = joblib.load(path)
        self.preprocessor = data['preprocessor']
        self.numeric_features = data['numeric_features']
        self.categorical_features = data['categorical_features']
        self.feature_names = data['feature_names']
        print(f"Preprocessor loaded from {path}")

def main():
    """Run preprocessing pipeline."""
    print("Loading featured data...")
    df = pd.read_csv('data/featured_invoices.csv')
    
    preprocessor = DataPreprocessor()
    
    print("\nIdentifying features...")
    preprocessor.identify_features(df, target_cols=['freight_cost', 'is_risky', 'risk_score'])
    
    print("\nPreparing regression data...")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocessor.prepare_data(
        df, target_col='freight_cost'
    )
    
    print("\nFitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train_reg)
    
    print(f"\nProcessed training shape: {X_train_processed.shape}")
    print(f"Feature names: {len(preprocessor.feature_names)} features")
    
    preprocessor.save_preprocessor()
    
    # Save splits for model training
    joblib.dump({
        'X_train': X_train_reg,
        'X_test': X_test_reg,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg
    }, 'data/train_test_splits.pkl')
    
    print("Train/test splits saved to data/train_test_splits.pkl")

if __name__ == '__main__':
    main()
