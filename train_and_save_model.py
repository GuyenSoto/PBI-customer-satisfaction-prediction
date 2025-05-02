# train_and_save_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import os

def train_save_model(input_file='ACME-HappinessSurvey2020.csv', output_dir='.'):
    """Train the model and save it for future use"""
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Identify target column
    if 'Y' in df.columns:
        target_col = 'Y'
    else:
        raise ValueError("Could not find column 'Y' in the dataset")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Analyze feature averages by class
    avg_by_class = {}
    for feature in X.columns:
        avg_happy = X[y == 1][feature].mean()
        avg_unhappy = X[y == 0][feature].mean()
        diff = avg_happy - avg_unhappy
        avg_by_class[feature] = (avg_happy, avg_unhappy, diff)
    
    # Store feature information
    feature_info = {
        'features': list(X.columns),
        'importance': [abs(avg_by_class[f][2]) for f in X.columns],
        'avg_happy': {f: avg_by_class[f][0] for f in X.columns},
        'avg_unhappy': {f: avg_by_class[f][1] for f in X.columns},
        'diff': {f: avg_by_class[f][2] for f in X.columns}
    }
    
    # Create ensemble model
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
    
    # Create and train voting ensemble
    print("Training voting ensemble model...")
    vote = VotingClassifier(estimators=estimators, voting='soft')
    vote.fit(X_scaled, y)
    
    # Save model, scaler and feature info
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, 'ensemble_model.pkl'), 'wb') as f:
        pickle.dump(vote, f)
    
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(output_dir, 'feature_info.pkl'), 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"Model and associated files saved to {output_dir}")
    print("Run the Streamlit app with: streamlit run prediction_app.py")

if __name__ == "__main__":
    train_save_model()