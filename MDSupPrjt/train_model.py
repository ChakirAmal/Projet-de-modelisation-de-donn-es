import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def build_model(X, layer_configurations, network_type, problem_type, num_classes=1):
    """
    Builds a neural network model based on the specified architecture.

    Parameters:
    - X (array): Input data to determine input shape.
    - layer_configurations (list): List of dictionaries with 'neurons' and 'activation' keys.
    - network_type (str): The type of neural network ('dense', 'cnn', or 'rnn').
    - task_type (str): Type of problem ('classification' or 'regression').
    - num_classes (int): Number of output classes for classification.

    Returns:
    - model (tf.keras.Model): Uncompiled neural network model.
    """
    model = tf.keras.Sequential()

    # Build layers based on network type
    if network_type == 'dense':
        for i, config in enumerate(layer_configurations):
            neurons = int(config['neurons'])
            activation = config['activation']
            if i == 0:
                model.add(tf.keras.layers.Dense(neurons, activation=activation, input_shape=(X.shape[1],)))
            else:
                model.add(tf.keras.layers.Dense(neurons, activation=activation))

    elif network_type == 'cnn':
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=layer_configurations[0]['activation'], input_shape=(X.shape[1], X.shape[2], 1)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        for config in layer_configurations[1:]:
            model.add(tf.keras.layers.Conv2D(config['neurons'], (3, 3), activation=config['activation']))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())

    # Add output layer based on task type
    if problem_type == 'classification':
        if num_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        else:
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    else:  # Regression
        model.add(tf.keras.layers.Dense(1, activation='linear'))

    return model

# Function to compile and train the model
def compile_and_train_model(model, X, y, problem_type, test_size=0.2, random_state=42, optimizer='adam', epochs=10, batch_size=32):
    # Compile the model
    if problem_type == 'classification':
        loss_function = 'binary_crossentropy'  # Change to 'categorical_crossentropy' for multiclass classification
        metrics = ['accuracy']
    else:
        loss_function = 'mean_squared_error'
        metrics = ['mae']

    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    # Train the model
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)



    return model, history

def save_model_and_history(model, history, file_name):
    model_hist_folder = 'model_hist'
    if not os.path.exists(model_hist_folder):
        os.makedirs(model_hist_folder)

    # Save the model in the new Keras format (.keras)
    model_save_path = os.path.join(model_hist_folder, f'{file_name}_model.keras')
    model.save(model_save_path)  # Use the new Keras format

    # Save the history as a CSV file
    history_file = os.path.join(model_hist_folder, f'history_{file_name}.csv')
    pd.DataFrame(history.history).to_csv(history_file)

    return model_save_path, history_file
