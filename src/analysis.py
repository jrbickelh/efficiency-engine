import shap
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def explain_model_shap(model, train_features, test_features, filename="shap_summary.png"):
    """
    Generates a SHAP summary plot to explain feature importance.
    Uses a KernelExplainer which works for any model (TF or XGB).
    """
    print(f"\nGeneratin SHAP explanation (saving to {filename})...")

    # We use a summary (kmeans) of training data to speed up the explanation
    # Checking 50 representative points is usually enough for a quick plot
    background_data = shap.kmeans(train_features, 50)

    # Create explainer
    # Note: For Keras models, we pass the model.predict function
    explainer = shap.KernelExplainer(model.predict, background_data)

    # Calculate SHAP values for a subset of test data (first 50 rows) to keep it fast
    shap_values = explainer.shap_values(test_features.iloc[:50])

    # Plot
    plt.figure()
    # reshape if shap returns a list (common with TF models)
    vals = shap_values[0] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(vals, test_features.iloc[:50], show=False)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_residuals(model, test_features, test_labels, filename="residuals.png"):
    """
    Plots Predicted vs Actual values.
    """
    predictions = model.predict(test_features).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(test_labels, predictions, alpha=0.6)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')

    # Plot diagonal line (Perfect prediction)
    lims = [0, 50]
    plt.plot(lims, lims, color='red')

    plt.title("Residual Analysis: True vs Predicted")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_correlation_matrix(dataset, filename="correlation_matrix.png"):
    """
    Plots the correlation between features to identify multicollinearity.
    """
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation
    numeric_data = dataset.select_dtypes(include=[np.number])
    corr = numeric_data.corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_error_distribution(model, features, labels, filename="error_distribution.png"):
    """
    Plots the histogram of prediction errors.
    Ideally, this should look like a Gaussian (Bell Curve) centered at 0.
    """
    predictions = model.predict(features).flatten()
    errors = labels - predictions

    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=25, kde=True, color='purple')
    plt.xlabel('Prediction Error [MPG]')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors (Residuals)')
    plt.axvline(x=0, color='k', linestyle='--', label='Zero Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()

def plot_physics_check(model, features, labels, feature_name='Weight', filename="physics_check.png"):
    """
    Visualizes how the model reacts to a specific physical feature compared to reality.
    Overlays predictions on top of actual data for a single variable.
    """
    plt.figure(figsize=(8, 6))

    # Plot Real Data
    plt.scatter(features[feature_name], labels, label='Actual Data', alpha=0.4, color='gray')

    # Plot Model Predictions
    predictions = model.predict(features).flatten()
    plt.scatter(features[feature_name], predictions, label='Model Predictions', alpha=0.6, color='blue', s=10)

    plt.xlabel(feature_name)
    plt.ylabel('MPG')
    plt.title(f'Physics Check: {feature_name} vs MPG')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
