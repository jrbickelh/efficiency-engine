import pandas as pd
import pandera as pa
from src.schema import AutoMPGSchema

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
COLUMN_NAMES = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

def load_and_engineer_data():
    """
    Fetches data, validates it against schema, and adds engineered features.
    """
    # 1. Load Raw Data
    raw_dataset = pd.read_csv(DATA_URL, names=COLUMN_NAMES,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

    dataset = raw_dataset.dropna().copy()

    # 2. Validate Data (Engineering)
    # We validate BEFORE one-hot encoding
    try:
        AutoMPGSchema.validate(dataset, lazy=True)
        print("✅ Data Schema Validation Passed")
    except pa.errors.SchemaErrors as err:
        print("❌ Data Schema Validation Failed:")
        print(err.failure_cases)
        raise

    # 3. Feature Engineering
    # Ratio of power to weight is physically significant for fuel efficiency
    dataset['Power_to_Weight'] = dataset['Horsepower'] / dataset['Weight']

    # 4. Preprocessing
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

    # Fix: Explicitly set dtype=int to avoid Booleans
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='', dtype=int)

    # Final Safety Net: Force entire dataset to float32
    # This prevents 'object' or 'bool' types from ever reaching TensorFlow
    dataset = dataset.astype('float32')

    return dataset

def split_data(dataset):
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    return train_features, test_features, train_labels, test_labels
