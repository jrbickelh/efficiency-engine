import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Custom Modules
from src.data_loader import load_and_engineer_data, split_data
from src.models import build_model
from src.visualization import plot_loss
from src.baselines import run_baseline_model
from src.analysis import (
    explain_model_shap,
    plot_residuals,
    plot_correlation_matrix,
    plot_error_distribution,
    plot_physics_check
    )

EPOCHS = 100

def main():
    print("=== Auto MPG Pipeline Started ===\n")

    # 1. Data Engineering
    print("[Step 1] Loading and Engineering Data...")
    dataset = load_and_engineer_data()
    train_features, test_features, train_labels, test_labels = split_data(dataset)
    print(f"Data ready: {len(train_features)} training samples.")

    # 2. Baseline Modeling (XGBoost)
    print("\n[Step 2] Benchmarking...")
    xgb_mae, xgb_model = run_baseline_model(train_features, train_labels, test_features, test_labels)
    print(f" > XGBoost Baseline MAE: {xgb_mae:.2f} MPG")

    # 3. Deep Learning (TensorFlow)
    print("\n[Step 3] Training Neural Network...")
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    dnn_model = build_model(normalizer, learning_rate=0.001, model_type='dnn')
    history = dnn_model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=0, epochs=EPOCHS
    )

    dnn_mae = dnn_model.evaluate(test_features, test_labels, verbose=0)
    print(f" > Deep Neural Net MAE:  {dnn_mae:.2f} MPG")

    # 4. Comparison
    print("\n[Step 4] Results Comparison")
    if dnn_mae < xgb_mae:
        print(f"🏆 The Neural Network won by {xgb_mae - dnn_mae:.2f} MPG!")
        winner = dnn_model
    else:
        print(f"🏆 The Baseline (XGBoost) won by {dnn_mae - xgb_mae:.2f} MPG!")
        winner = xgb_model # Note: SHAP explanation for XGBoost handles differently,
                           # so for simplicity we will explain the DNN below.

    # 5. Analysis & Visualization
    print("\n[Step 5] Generating Analysis Artifacts...")

    # A. Data Analysis (EDA)
    # We use the full dataset for correlation analysis before splitting
    plot_correlation_matrix(dataset, filename="correlation_matrix.png")

    # B. Model Performance
    plot_loss(history, "DNN Training History", filename="training_curve.png")
    plot_residuals(dnn_model, test_features, test_labels, filename="residuals.png")
    plot_error_distribution(dnn_model, test_features, test_labels, filename="error_distribution.png")

    # C. Physics/Behavior Check
    # We check 'Weight' because it has the strongest physical relationship with MPG
    plot_physics_check(dnn_model, test_features, test_labels, feature_name='Weight', filename="physics_check.png")

    # D. Explainability
    explain_model_shap(dnn_model, train_features, test_features, filename="shap_explanation.png")

    print("\n✅ Pipeline Complete. Check your folder for all 6 visualizations.")

if __name__ == "__main__":
    main()
