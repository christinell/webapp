import numpy as np
import random
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pickle

# Define categorical features list
categorical_features = ['Sex', 'Residence', 'EL', 'EF', 'DP', 'SS', 'AI', 'HS', 'T2DM', 'FHTN']

def optimize_parameters(data_train, max_evals=20):
    """Optimize CatBoost classifier parameters using Bayesian optimization.
    
    Args:
        data_train: Training DataFrame containing features and 'HTN' target
        max_evals: Maximum number of optimization evaluations
    
    Returns:
        Tuple of (best_params, best_hyperparameters, trials_object)
    """
    
    # Base parameters configuration
    base_params = {
        'eval_metric': 'AUC',
        'class_weights': [1, 5.6],  # Handling class imbalance
        'cat_features': categorical_features,
        'thread_count': 20,
        'logging_level': 'Silent',
        'random_seed': 3008,  # Fixed seed for reproducibility
        'allow_writing_files': False  # Disable output files
    }

    # Define hyperparameter search space
    hyperopt_space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'depth': hp.quniform('depth', 4, 10, 1),  # More conservative depth range
        'iterations': hp.quniform('iterations', 500, 2000, 100),  # Higher minimum iterations
        'rsm': hp.uniform('rsm', 0.6, 1.0),  # Feature sampling ratio
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(2), np.log(30)),
        'bootstrap_type': hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'random_strength': hp.loguniform('random_strength', np.log(1e-9), np.log(1)),
        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
        'subsample': hp.uniform('subsample', 0.6, 1.0)  # More conservative subsampling
    }

    # Track best score during optimization
    best_score = {'value': 0}

    def objective(hyper_params):
        """Objective function for hyperparameter optimization."""
        
        # Convert parameters to appropriate types
        params = base_params.copy()
        params.update({
            'learning_rate': hyper_params['learning_rate'],
            'depth': int(hyper_params['depth']),
            'iterations': int(hyper_params['iterations']),
            'rsm': hyper_params['rsm'],
            'l2_leaf_reg': hyper_params['l2_leaf_reg'],
            'bootstrap_type': hyper_params['bootstrap_type'],
            'random_strength': hyper_params['random_strength']
        })

        # Handle bootstrap-specific parameters
        if hyper_params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = hyper_params['bagging_temperature']
            params.pop('subsample', None)  # Ensure subsample is removed
        else:
            params['subsample'] = hyper_params['subsample']
            params.pop('bagging_temperature', None)  # Ensure bagging_temp is removed

        # Prepare data
        X = data_train.drop(columns=['HTN'])
        y = data_train['HTN']

        # Perform cross-validation
        mean_auc = cross_val_score(regressor, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True),
                                scoring='roc_auc').mean()

        # Update best model if improved
        if mean_auc > best_score['value']:
            best_score['value'] = mean_auc
            # Save parameters
            with open('catboost/catboost_best.pkl', 'wb') as f:
                pickle.dump((params, mean_auc), f)

        return {'loss': -mean_auc, 'status': STATUS_OK}

    # Run optimization
    trials = Trials()
    best_hyperparams = fmin(fn=objective,
                            space=hyperopt_space,
                            algo=tpe.suggest,
                            max_evals=max_evals,
                            trials=trials
                            )

    return base_params, best_hyperparams, trials
