import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

class ModelExplainer:
    """SHAP-based model explainability for invoice predictions."""
    
    def __init__(self):
        self.explainer = None
        self.shap_values = None
    
    def explain_regressor(self, model, X, feature_names=None):
        """Generate SHAP explanations for regression model."""
        
        print("Generating SHAP explanations for regression model...")
        
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X)
        
        if feature_names is not None:
            self.feature_names = feature_names
        
        return self.shap_values
    
    def plot_summary(self, X, save_path='outputs/shap_summary.png'):
        """Plot SHAP summary plot."""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, 
                         feature_names=self.feature_names if hasattr(self, 'feature_names') else None,
                         show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plot saved to {save_path}")
    
    def plot_bar(self, X, save_path='outputs/shap_bar.png'):
        """Plot SHAP feature importance bar chart."""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X,
                         feature_names=self.feature_names if hasattr(self, 'feature_names') else None,
                         plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Bar plot saved to {save_path}")
    
    def explain_single_prediction(self, model, X_sample, feature_names=None):
        """Explain a single prediction."""
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        explanation = {
            'base_value': explainer.expected_value,
            'prediction': model.predict(X_sample.reshape(1, -1))[0],
            'feature_importance': {}
        }
        
        if feature_names is not None:
            for i, (name, value) in enumerate(zip(feature_names, shap_values[0])):
                explanation['feature_importance'][name] = float(value)
        
        return explanation
    
    def get_top_features(self, n=10):
        """Get top N most important features."""
        
        if self.shap_values is None:
            return []
        
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-n:][::-1]
        
        if hasattr(self, 'feature_names'):
            return [(self.feature_names[i], mean_abs_shap[i]) for i in top_indices]
        else:
            return [(f'Feature_{i}', mean_abs_shap[i]) for i in top_indices]

def main():
    """Run explainability analysis."""
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("Loading models and data...")
    regressor = joblib.load('models/freight_regressor.pkl')
    
    # Load preprocessed data
    data = joblib.load('data/train_test_splits.pkl')
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.load_preprocessor('models/preprocessor.pkl')
    
    X_test = data['X_test']
    X_test_processed = preprocessor.transform(X_test)
    
    explainer = ModelExplainer()
    
    print("\nGenerating SHAP explanations...")
    shap_values = explainer.explain_regressor(regressor, X_test_processed, preprocessor.feature_names)
    
    print("\nCreating visualizations...")
    explainer.plot_summary(X_test_processed)
    explainer.plot_bar(X_test_processed)
    
    print("\nTop 10 Features:")
    for name, importance in explainer.get_top_features(10):
        print(f"  {name}: {importance:.4f}")
    
    print("\nExplainability analysis completed!")

if __name__ == '__main__':
    main()
