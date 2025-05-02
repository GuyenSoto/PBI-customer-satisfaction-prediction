#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Ensemble Model for Customer Happiness Prediction
Uses voting ensemble techniques to achieve >80% accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Models and utilities
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Base models
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Configuration
np.random.seed(42)
input_file = 'ACME-HappinessSurvey2020.csv'
output_dir = r'd:\Guyen\Jobs\Examples\PBIX_Git\1Project'

def load_data():
    """Load and prepare the data"""
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify target column
    if 'Y' in df.columns:
        target_col = 'Y'
    else:
        raise ValueError("Could not find column 'Y' in the dataset")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return df, X, y

def train_voting_ensemble(X, y):
    """Train an optimized voting ensemble model"""
    print("\n--- TRAINING OPTIMIZED VOTING ENSEMBLE ---")
    
    # Create multiple models with different configurations
    estimators = [
        ('rf1', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ('rf2', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=2, random_state=24)),
        ('gb1', GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ('gb2', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=24)),
        ('lgbm1', LGBMClassifier(n_estimators=200, num_leaves=31, random_state=42, verbosity=-1)),
        ('lgbm2', LGBMClassifier(n_estimators=100, num_leaves=20, max_depth=5, random_state=24, verbosity=-1)),
        ('svm', SVC(probability=True, kernel='rbf', C=1.0, random_state=42)),
        ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ]
    
    # Create soft voting ensemble (using probabilities)
    vote = VotingClassifier(estimators=estimators, voting='soft')
    
    # Train the model
    print("Training ensemble with 8 base models...")
    vote.fit(X, y)
    
    # Evaluate the model
    y_pred = vote.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"Voting ensemble accuracy: {accuracy:.4f}")
    
    return vote, accuracy

def apply_feature_scaling(X):
    """Apply feature scaling to improve convergence"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def analyze_data(X, y):
    """Analyze the dataset to better understand relationships"""
    print("\n--- DATA ANALYSIS ---")
    
    # Check class balance
    class_counts = y.value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    # Correlation between features
    corr_matrix = X.corr()
    
    # Most correlated feature pairs
    high_corr_pairs = []
    for i, col1 in enumerate(X.columns):
        for col2 in X.columns[i+1:]:
            corr = corr_matrix.loc[col1, col2]
            if abs(corr) > 0.5:  # Arbitrary threshold
                high_corr_pairs.append((col1, col2, corr))
    
    if high_corr_pairs:
        print("Highly correlated feature pairs:")
        for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {col1} - {col2}: {corr:.4f}")
    else:
        print("No highly correlated features found.")
    
    # Calculate averages by class
    avg_by_class = {}
    for feature in X.columns:
        avg_happy = X[y == 1][feature].mean()
        avg_unhappy = X[y == 0][feature].mean()
        diff = avg_happy - avg_unhappy
        avg_by_class[feature] = (avg_happy, avg_unhappy, diff)
    
    print("\nAverage differences by class:")
    for feature, (avg_happy, avg_unhappy, diff) in sorted(avg_by_class.items(), key=lambda x: abs(x[1][2]), reverse=True):
        print(f"  {feature}: Happy = {avg_happy:.2f}, Unhappy = {avg_unhappy:.2f}, Diff = {diff:.2f}")
    
    return avg_by_class

def optimize_model(df, X, y):
    """Optimize the model for maximum accuracy"""
    
    # 1. Analyze data
    avg_by_class = analyze_data(X, y)
    
    # 2. Scale features
    print("\nApplying feature scaling...")
    X_scaled = apply_feature_scaling(X)
    
    # 3. Create additional features to improve performance
    print("\nCreating additional features...")
    X_enhanced = X_scaled.copy()
    
    # Add interactions between features with the largest differences
    # Sort features by absolute difference
    features_by_diff = sorted(avg_by_class.items(), key=lambda x: abs(x[1][2]), reverse=True)
    top_features = [f[0] for f in features_by_diff[:3]]  # Top 3 features
    
    print(f"Most discriminative features: {top_features}")
    
    # Add interactions
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            feat_name = f"{feat1}_x_{feat2}"
            X_enhanced[feat_name] = X_scaled[feat1] * X_scaled[feat2]
    
    # 4. Train ensemble model
    model, accuracy = train_voting_ensemble(X_enhanced, y)
    
    # 5. Generate predictions
    df_with_predictions = df.copy()
    df_with_predictions['Prediction'] = model.predict(X_enhanced)
    df_with_predictions['Probability'] = model.predict_proba(X_enhanced)[:, 1]
    
    # 6. Calculate confusion matrix
    cm = confusion_matrix(y, df_with_predictions['Prediction'])
    
    # 7. Prepare and return results
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': df_with_predictions,
        'confusion_matrix': cm,
        'X': X,
        'X_enhanced': X_enhanced,
        'y': y,
        'avg_by_class': avg_by_class,
        'original_data': df
    }

def export_results(results, output_dir):
    """Export all results to files"""
    print("\n--- EXPORTING RESULTS ---")
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save all results as CSV to avoid any conversion issues
    
    # 1. Main dataset with predictions
    predictions_path = os.path.join(output_dir, "predictions.csv")
    results['predictions'].to_csv(predictions_path, index=False)
    print(f"Exported main dataset to {predictions_path}")
    
    # Try to export as parquet as well
    try:
        parquet_path = os.path.join(output_dir, "predictions.parquet")
        results['predictions'].to_parquet(parquet_path, index=False)
        print(f"Also exported as parquet to {parquet_path}")
    except Exception as e:
        print(f"Could not export as parquet: {e}")
    
    # 2. Confusion matrix
    cm = results['confusion_matrix']
    confusion_data = pd.DataFrame({
        'Actual': ['Unhappy', 'Unhappy', 'Happy', 'Happy'],
        'Predicted': ['Unhappy', 'Happy', 'Unhappy', 'Happy'],
        'Count': [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]]
    })
    confusion_path = os.path.join(output_dir, "confusion_matrix.csv")
    confusion_data.to_csv(confusion_path, index=False)
    print(f"Exported confusion matrix to {confusion_path}")
    
    # Try to export as parquet as well
    try:
        parquet_path = os.path.join(output_dir, "confusion_matrix.parquet")
        confusion_data.to_parquet(parquet_path, index=False)
        print(f"Also exported as parquet to {parquet_path}")
    except Exception as e:
        print(f"Could not export as parquet: {e}")
    
    # 3. Metrics
    metrics_data = pd.DataFrame({
        'Metric': ['Accuracy'],
        'Value': [results['accuracy']]
    })
    metrics_path = os.path.join(output_dir, "metrics.csv")
    metrics_data.to_csv(metrics_path, index=False)
    print(f"Exported metrics to {metrics_path}")
    
    # Try to export as parquet as well
    try:
        parquet_path = os.path.join(output_dir, "metrics.parquet")
        metrics_data.to_parquet(parquet_path, index=False)
        print(f"Also exported as parquet to {parquet_path}")
    except Exception as e:
        print(f"Could not export as parquet: {e}")
    
    # 4. Model name (separate file to avoid type issues)
    model_data = pd.DataFrame({
        'Metric': ['Model'],
        'Value': ['Voting Ensemble']
    })
    model_path = os.path.join(output_dir, "model_name.csv")
    model_data.to_csv(model_path, index=False)
    print(f"Exported model name to {model_path}")
    
    # 5. Feature importance (using average differences)
    importance_data = []
    for feature, (avg_happy, avg_unhappy, diff) in results['avg_by_class'].items():
        # Only include original features (X1-X6)
        if feature in results['X'].columns:
            importance_data.append({
                'feature': feature,
                'importance': abs(diff)  # Use absolute difference as importance
            })
    
    feature_importance = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
    importance_path = os.path.join(output_dir, "feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    print(f"Exported feature importance to {importance_path}")
    
    # Try to export as parquet as well
    try:
        parquet_path = os.path.join(output_dir, "feature_importance.parquet")
        feature_importance.to_parquet(parquet_path, index=False)
        print(f"Also exported as parquet to {parquet_path}")
    except Exception as e:
        print(f"Could not export as parquet: {e}")
    
    # 6. Average values by feature
    avg_data = []
    for feature, (avg_happy, avg_unhappy, diff) in results['avg_by_class'].items():
        # Only include original features (X1-X6)
        if feature in results['X'].columns:
            avg_data.append({
                'Feature': feature,
                'Avg_Happy': avg_happy,
                'Avg_Unhappy': avg_unhappy,
                'Difference': diff
            })
    
    averages_data = pd.DataFrame(avg_data)
    averages_path = os.path.join(output_dir, "feature_averages.csv")
    averages_data.to_csv(averages_path, index=False)
    print(f"Exported feature averages to {averages_path}")
    
    # Try to export as parquet as well
    try:
        parquet_path = os.path.join(output_dir, "feature_averages.parquet")
        averages_data.to_parquet(parquet_path, index=False)
        print(f"Also exported as parquet to {parquet_path}")
    except Exception as e:
        print(f"Could not export as parquet: {e}")

def generate_visualizations(results, output_dir):
    """Generate visualizations for analysis"""
    print("\n--- GENERATING VISUALIZATIONS ---")
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unhappy', 'Happy'],
                yticklabels=['Unhappy', 'Happy'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # 2. Feature importance
    importance_data = []
    for feature, (avg_happy, avg_unhappy, diff) in results['avg_by_class'].items():
        # Only include original features (X1-X6)
        if feature in results['X'].columns:
            importance_data.append({
                'feature': feature,
                'importance': abs(diff)  # Use absolute difference as importance
            })
    
    feature_importance = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    
    # 3. Average difference by feature
    avg_data = []
    for feature, (avg_happy, avg_unhappy, diff) in results['avg_by_class'].items():
        # Only include original features (X1-X6)
        if feature in results['X'].columns:
            avg_data.append({
                'Feature': feature,
                'Difference': diff
            })
    
    diff_data = pd.DataFrame(avg_data).sort_values('Difference', ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Difference', y='Feature', data=diff_data)
    plt.title('Average Difference Between Happy and Unhappy Customers')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_difference.png"))
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function"""
    print("=== ADVANCED ENSEMBLE MODEL FOR CUSTOMER HAPPINESS PREDICTION ===")
    
    # 1. Load data
    df, X, y = load_data()
    
    # 2. Optimize model
    results = optimize_model(df, X, y)
    
    # 3. Export results (both CSV and parquet)
    export_results(results, output_dir)
    
    # 4. Generate visualizations
    generate_visualizations(results, output_dir)
    
    print("\n=== PROCESS COMPLETED ===")
    print(f"Final model accuracy: {results['accuracy']:.4f}")
    print(f"All files exported to: {output_dir}")
    
    # More detailed information on classification
    cm = results['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    print(f"\nConfusion matrix:")
    print(f"  True negatives: {tn} ({tn/total:.1%})")
    print(f"  False positives: {fp} ({fp/total:.1%})")
    print(f"  False negatives: {fn} ({fn/total:.1%})")
    print(f"  True positives: {tp} ({tp/total:.1%})")
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAdditional metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
if __name__ == "__main__":
    main()
