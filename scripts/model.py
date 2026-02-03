"""LightGBM training utilities.

Functions:
- prepare_lgb_data(df, features, target='BID') -> X, Y, categorical_features, numerical_features
- train_lgb_cv(X, Y, categorical_features, params=None, n_splits=4, manual_threshold=0.12)

"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score


def prepare_lgb_data(df: pd.DataFrame, features: list, target: str = 'BID'):
    df = df.copy()
    X = df[features].copy()
    Y = df[target].copy()

    categorical_features = [col for col in features if X[col].dtype == 'object' or X[col].dtype.name == 'category']
    numerical_features = [col for col in features if col not in categorical_features]

    # Label encode categorical columns
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X[col] = X[col].astype('category')

    for col in numerical_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)

    return X, Y, categorical_features, numerical_features


def train_lgb_cv(X, Y, categorical_features, params=None, n_splits=4, manual_threshold=0.12):
    if params is None:
        params = dict(objective='binary', metric='auc', random_state=42,
                      n_estimators=317, learning_rate=0.03, num_leaves=31)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    auc_scores = []
    f1_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        Y_train, Y_val = Y.iloc[train_idx], Y.iloc[val_idx]

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, Y_train,
                eval_set=[(X_val, Y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(100, verbose=False)],
                categorical_feature=[c for c in categorical_features if c in X_train.columns]
               )

        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= manual_threshold).astype(int)

        auc = roc_auc_score(Y_val, y_pred_proba)
        f1 = f1_score(Y_val, y_pred)

        auc_scores.append(auc)
        f1_scores.append(f1)
        models.append(clf)

    metrics = {
        'auc_mean': np.mean(auc_scores),
        'auc_std': np.std(auc_scores),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores),
        'fold_aucs': auc_scores,
        'fold_f1s': f1_scores
    }

    return models, metrics
