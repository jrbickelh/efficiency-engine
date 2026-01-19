import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(normalizer, learning_rate=0.001, model_type='dnn'):
    """
    Builds and compiles a Keras model.
    
    Args:
        normalizer (layer): The normalization layer adapted to the training data.
        learning_rate (float): Step size for the optimizer.
        model_type (str): 'linear' for simple regression, 'dnn' for deep neural net.
    """
    if model_type == 'linear':
        model = keras.Sequential([
            normalizer,
            layers.Dense(units=1)
        ])
    else:
        model = keras.Sequential([
            normalizer,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_absolute_error'
    )
    
    return model
