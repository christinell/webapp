from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score)
import _pickle as pickle
from typing import Tuple, Dict, Any

def train_and_evaluate_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_params: Dict[str, Any]
) -> Tuple[CatBoostClassifier, Dict[str, float]]:
    """Train and evaluate a CatBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_params: Parameters for CatBoostClassifier
        
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    # Initialize and train the model
    model = CatBoostClassifier(**model_params)
    model.fit(X_train, y_train, verbose=False)
    
    # Generate predictions
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    metrics = {
        'roc_auc': roc_auc_score(y_test, test_prob),
        'average_precision': average_precision_score(y_test, test_prob),
        'accuracy': accuracy_score(y_test, test_pred),
        'precision': precision_score(y_test, test_pred),
        'recall': recall_score(y_test, test_pred)
    }
    
    return model, metrics

def save_model_artifacts(
    model: CatBoostClassifier,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    file_path: str = 'catboost_artifacts.pkl'
) -> None:
    """Save model artifacts to a pickle file.
    
    Args:
        model: Trained CatBoost model
        params: Model parameters
        metrics: Evaluation metrics
        file_path: Path to save the artifacts
    """
    artifacts = {
        'model': model,
        'params': params,
        'metrics': metrics
    }
    
    with open(file_path, 'wb') as f:
        pickle.dump(artifacts, f)

def load_model_artifacts(
    file_path: str = 'catboost_artifacts.pkl'
) -> Tuple[CatBoostClassifier, Dict[str, Any], Dict[str, float]]:
    """Load saved model artifacts.
    
    Args:
        file_path: Path to saved artifacts file
        
    Returns:
        Tuple of (model, params, metrics)
    """
    with open(file_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    return artifacts['model'], artifacts['params'], artifacts['metrics']

def main() -> None:
    """Main execution workflow."""
    # Load and prepare data
    data = pd.read_csv('your_data.csv')  # Replace with actual data path
    X = data.drop(columns=['HTN'])
    y = data['HTN']
    
    # Split data
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train['HTN'] = y_train
    # Hyperparameter optimization
    optimize_parameters(X_train, max_evals=1000)
    
    # Train and evaluate model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    with open('catboost/catboost.pkl','rb') as f:
        params,score = pickle.load(f)
    model, metrics = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, params
    )
    
    # Save model artifacts
    save_model_artifacts(model, optimized_params, metrics)



if __name__ == "__main__":
    main()