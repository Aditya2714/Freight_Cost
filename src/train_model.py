import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
import joblib
import os
import json

class ModelTrainer:
    """Train and evaluate ML models for freight cost prediction and risk detection."""
    
    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.metrics = {}
    
    def train_regressor(self, X_train, y_train, optimize=True):
        """Train Random Forest Regressor for freight cost prediction."""
        
        print("Training Regression Model (Freight Cost Prediction)...")
        
        if optimize:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', 
                                      n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            self.regressor = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.regressor = RandomForestRegressor(n_estimators=200, max_depth=20, 
                                                    random_state=42, n_jobs=-1)
            self.regressor.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.regressor, X_train, y_train, cv=5, 
                                    scoring='neg_mean_absolute_error')
        print(f"CV MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
        
        return self.regressor
    
    def train_classifier(self, X_train, y_train, optimize=True):
        """Train Random Forest Classifier for risk detection."""
        
        print("\nTraining Classification Model (Risk Detection)...")
        
        if optimize:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced', None]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', 
                                      n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            self.classifier = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.classifier = RandomForestClassifier(n_estimators=200, max_depth=20,
                                                      class_weight='balanced', random_state=42)
            self.classifier.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.classifier, X_train, y_train, cv=5, scoring='f1')
        print(f"CV F1-Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        return self.classifier
    
    def evaluate_regressor(self, X_test, y_test):
        """Evaluate regression model."""
        
        y_pred = self.regressor.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        self.metrics['regression'] = metrics
        
        print("\nRegression Model Evaluation:")
        print(f"  MAE:  ${metrics['mae']:.2f}")
        print(f"  RMSE: ${metrics['rmse']:.2f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics, y_pred
    
    def evaluate_classifier(self, X_test, y_test):
        """Evaluate classification model."""
        
        y_pred = self.classifier.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.metrics['classification'] = metrics
        
        print("\nClassification Model Evaluation:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
        
        return metrics, y_pred
    
    def save_models(self, path='models/'):
        """Save trained models to disk."""
        os.makedirs(path, exist_ok=True)
        
        joblib.dump(self.regressor, os.path.join(path, 'freight_regressor.pkl'))
        joblib.dump(self.classifier, os.path.join(path, 'risk_classifier.pkl'))
        
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\nModels saved to {path}")
    
    def load_models(self, path='models/'):
        """Load trained models from disk."""
        self.regressor = joblib.load(os.path.join(path, 'freight_regressor.pkl'))
        self.classifier = joblib.load(os.path.join(path, 'risk_classifier.pkl'))
        
        with open(os.path.join(path, 'metrics.json'), 'r') as f:
            self.metrics = json.load(f)
        
        print(f"Models loaded from {path}")

def main():
    """Run model training pipeline."""
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import DataPreprocessor
    
    print("Loading data...")
    df = pd.read_csv('data/featured_invoices.csv')
    
    preprocessor = DataPreprocessor()
    preprocessor.identify_features(df, target_cols=['freight_cost', 'is_risky', 'risk_score'])
    
    # Prepare regression data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = preprocessor.prepare_data(
        df, target_col='freight_cost'
    )
    
    # Prepare classification data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = preprocessor.prepare_data(
        df, target_col='is_risky'
    )
    
    # Fit preprocessor on regression training data
    X_train_reg_processed = preprocessor.fit_transform(X_train_reg)
    X_test_reg_processed = preprocessor.transform(X_test_reg)
    
    # Transform classification data with same preprocessor
    X_train_clf_processed = preprocessor.transform(X_train_clf)
    X_test_clf_processed = preprocessor.transform(X_test_clf)
    
    # Train models
    trainer = ModelTrainer()
    
    trainer.train_regressor(X_train_reg_processed, y_train_reg, optimize=False)
    trainer.train_classifier(X_train_clf_processed, y_train_clf, optimize=False)
    
    # Evaluate models
    reg_metrics, reg_preds = trainer.evaluate_regressor(X_test_reg_processed, y_test_reg)
    clf_metrics, clf_preds = trainer.evaluate_classifier(X_test_clf_processed, y_test_clf)
    
    # Save models and preprocessor
    trainer.save_models()
    preprocessor.save_preprocessor()
    
    print("\nTraining pipeline completed!")

if __name__ == '__main__':
    main()
