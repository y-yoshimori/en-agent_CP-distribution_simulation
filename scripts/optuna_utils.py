"""Optuna helper for LightGBM hyperparameter search."""

import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np


def optuna_search(X, Y, categorical_features, n_trials=50, n_splits=4, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'random_state': random_state,
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'max_depth': trial.suggest_int('max_depth', 2, 64),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        }

        oof_preds = np.zeros(len(X))

        for train_idx, val_idx in skf.split(X, Y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = Y.iloc[train_idx], Y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      eval_metric='auc',
                      callbacks=[lgb.early_stopping(100, verbose=False)],
                      categorical_feature=categorical_features)

            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        return roc_auc_score(Y, oof_preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study
