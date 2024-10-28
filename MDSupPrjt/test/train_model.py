import os
import pandas as pd
import tensorflow as tf

# Function to build the model based on user configuration
def build_model(X, num_layers, layer_configurations, task_type):
    model = tf.keras.Sequential()

    # Add layers dynamically based on user input
    for i, config in enumerate(layer_configurations):
        neurons = config['neurons']
        activation = config['activation']

        if i == 0:
            # Add the first layer with input shape
            model.add(tf.keras.layers.Dense(neurons, activation=activation, input_shape=(X.shape[1],)))
        else:
            # Add hidden layers
            model.add(tf.keras.layers.Dense(neurons, activation=activation))

    # Output layer and loss function configuration based on task type
    if task_type == 'classification':
        output_activation = 'sigmoid'  # Use 'softmax' for multiclass classification
        model.add(tf.keras.layers.Dense(1, activation=output_activation))  # Single output for binary classification
    else:  # Regression task
        output_activation = 'linear'
        model.add(tf.keras.layers.Dense(1, activation=output_activation))  # Single output for regression

    return model

# Function to compile and train the model
def compile_and_train_model(model, X, y, task_type, epochs=10, batch_size=32):
    # Compile the model
    if task_type == 'classification':
        loss_function = 'binary_crossentropy'  # Change to 'categorical_crossentropy' for multiclass classification
        metrics = ['accuracy']
    else:
        loss_function = 'mean_squared_error'
        metrics = ['mae']

    model.compile(optimizer='adam', loss=loss_function, metrics=metrics)

    # Train the model
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, history

# Function to save the model and training history
def save_model_and_history(model, history, file_name):
    model_hist_folder = 'model_hist'
    if not os.path.exists(model_hist_folder):
        os.makedirs(model_hist_folder)

    model_save_path = os.path.join(model_hist_folder, f'{file_name}_model.h5')
    model.save(model_save_path)

    history_file = os.path.join(model_hist_folder, f'history_{file_name}.csv')
    pd.DataFrame(history.history).to_csv(history_file)

    return model_save_path, history_file
