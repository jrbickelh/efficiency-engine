from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def run_baseline_model(train_features, train_labels, test_features, test_labels):
    """
    Trains an XGBoost Regressor as a baseline comparison.
    """
    print("\n--- Training Baseline (XGBoost) ---")

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(train_features, train_labels)

    predictions = xgb_model.predict(test_features)
    mae = mean_absolute_error(test_labels, predictions)

    return mae, xgb_model
