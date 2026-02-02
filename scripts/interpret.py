import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(models, X, top_n=15):
    """Plot average feature importances from trained LightGBM model(s).

    models: single estimator or list of estimators (with attribute feature_importances_)
    X: DataFrame used for feature names
    """
    if isinstance(models, (list, tuple)):
        importances = np.mean([m.feature_importances_ for m in models], axis=0)
    else:
        importances = models.feature_importances_

    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, min(0.5 * top_n, 8)))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title('LightGBM Feature Importance (Top {})'.format(top_n))
    plt.tight_layout()
    plt.show()

    return importance_df


def compute_shap_values(model, X_explain, nsamples=100):
    """Compute SHAP values for a model and dataset.

    model: a trained LightGBM model (or first model from an ensemble)
    X_explain: DataFrame to explain
    """
    try:
        import shap
    except Exception as e:
        raise ImportError('shap is required for SHAP explanations: pip install shap') from e

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    return explainer, shap_values


def plot_shap_summary(shap_values, X_explain, plot_type='dot'):
    import shap
    import matplotlib.pyplot as plt

    if isinstance(shap_values, (list, tuple)):
        # For binary classification, shap_values[1] is usually the positive class
        vals = shap_values[1]
    else:
        vals = shap_values

    if plot_type == 'bar':
        shap.summary_plot(vals, X_explain, plot_type='bar', show=False)
    else:
        shap.summary_plot(vals, X_explain, show=False)

    plt.tight_layout()
    plt.show()


def plot_shap_dependence(shap_values, X_explain, feature_name, interaction_index=None):
    import shap
    import matplotlib.pyplot as plt

    if isinstance(shap_values, (list, tuple)):
        vals = shap_values[1]
    else:
        vals = shap_values

    shap.dependence_plot(feature_name, vals, X_explain, interaction_index=interaction_index, show=False)
    plt.tight_layout()
    plt.show()
