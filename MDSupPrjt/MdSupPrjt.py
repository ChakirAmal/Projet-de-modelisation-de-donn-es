from flask import Flask, render_template, send_from_directory, request, redirect, url_for, flash, jsonify
import os
import pandas as pd
import io
import base64
import graphviz
from train_model import build_model, compile_and_train_model, save_model_and_history  # Import training functions
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'ADMIN'

# Ensure 'upload' directory exists
UPLOAD_FOLDER = 'upload'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    files = os.listdir(UPLOAD_FOLDER) if os.path.exists(UPLOAD_FOLDER) else []
    return render_template('Interface.html', files=files)

@app.route('/MDSupPrjt/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        flash('Fichier uploadé avec succès!')
        return redirect(url_for('index'))
    
    flash('Erreur lors de l\'upload du fichier.')
    return redirect(url_for('index'))

@app.route('/MDSupPrjt/model_config')
def model_config():
    file_name = request.args.get('file_name')
    file_path = os.path.join(UPLOAD_FOLDER, file_name)

    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        columns = data.columns.tolist()  # Get the list of columns
    else:
        columns = []
    
    # Debugging: Print the columns found in the dataset
    print("Columns found:", columns)

    return render_template('model_config.html', file_name=file_name, columns=columns)

@app.route('/MDSupPrjt/generate_network', methods=['POST'])
def generate_network():
    problem_type = request.form.get('problem_type')
    num_layers = int(request.form.get('layers'))

    # Gather layer configurations
    layer_configurations = []
    for i in range(1, num_layers + 1):
        neurons = int(request.form.get(f'neurons_{i}'))
        activation = request.form.get(f'activation_{i}') if i < 3 else None
        layer_configurations.append({
            'neurons': neurons,
            'activation': activation
        })

    # Generate the network visualization
    dot = graphviz.Digraph()

    # Input layer
    with dot.subgraph(name='cluster_input') as c:
        for i in range(layer_configurations[0]['neurons']):
            c.node(f'I{i}', f'Input {i+1}')
        c.attr(label='Input Layer')

    # Add hidden layers
    for i, layer in enumerate(layer_configurations):
        with dot.subgraph(name=f'cluster_{i+1}') as c:
            for j in range(layer['neurons']):
                c.node(f'H{i+1}{j}', f'Neuron {j+1}')
            c.attr(label=f'Hidden Layer {i+1}')
        
        # Connect layers
        if i == 0:
            for k in range(layer['neurons']):
                dot.edge(f'I{k}', f'H{i+1}{k}')
        else:
            for m in range(layer_configurations[i-1]['neurons']):
                for n in range(layer['neurons']):
                    dot.edge(f'H{i}{m}', f'H{i+1}{n}')

    # Output layer
    output_neurons = 1
    with dot.subgraph(name='cluster_output') as c:
        for i in range(output_neurons):
            c.node(f'O{i}', f'Output {i+1}')
        c.attr(label='Output Layer')

    for i in range(layer_configurations[-1]['neurons']):
        for j in range(output_neurons):
            dot.edge(f'H{num_layers}{i}', f'O{j}')

    # Save graph to image
    img_stream = io.BytesIO()
    img_data = dot.pipe(format='png')
    img_stream.write(img_data)
    img_stream.seek(0)

    # Convert to base64 to send to frontend
    base64_img = base64.b64encode(img_stream.getvalue()).decode('utf8')

    return jsonify({'image_data': base64_img})

@app.route('/MDSupPrjt/visualize/<file_name>')
def visualize_results(file_name):
    # feature_x = request.args.get('feature_x')
    # feature_y = request.args.get('feature_y')

    # file_path = os.path.join(UPLOAD_FOLDER, file_name)
    # data = pd.read_csv(file_path)

    # if feature_x not in data.columns or feature_y not in data.columns:
    #     flash(f'Le fichier ne contient pas de colonnes "{feature_x}" et "{feature_y}".')
    #     return redirect(url_for('index'))

    # # Create a scatter plot based on the specified features and labels
    # plt.figure(figsize=(10, 6))
    # plt.scatter(data[feature_x], data[feature_y], c=data['label'], cmap='viridis')
    # plt.xlabel(feature_x)
    # plt.ylabel(feature_y)
    # plt.title('Visualisation des Données')

    # img = io.BytesIO()
    # plt.savefig(img, format='png')
    # img.seek(0)
    # plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # return render_template('visualize.html', data=data.to_html(), plot_url=plot_url)
    return

@app.route('/MDSupPrjt/train_model_route', methods=['POST'])
def train_model_route():
    training_data = request.get_json()

    # Print the received data for debugging
    print("Received training data:", training_data)

    # Extract parameters from the request
    dataset_name = training_data.get('dataset')
    x_features = training_data.get('features')
    y_feature = training_data.get('label')
    problem_type = training_data.get('problem_type')
    optimizer = training_data.get('optimizer')
    loss_function = training_data.get('loss_function')
    epochs = int(training_data.get('epochs'))
    num_class=int(training_data.get('num_class'))
    layer_configurations = training_data.get('layers')
    network_type = training_data.get('network_type')
    # Load the dataset
    file_path = os.path.join(UPLOAD_FOLDER, dataset_name)
    df = pd.read_csv(file_path)

    print("Dataframe Head:\n", df.head())  # This will print the first few rows of the dataframe

    # Extract features (X) and target (Y)
    X = df[x_features].values  # X is your feature columns
    y = df[y_feature].values   # y is your label/target column

    # Print the first few rows of the X and y arrays for debugging
    print("First 5 rows of X (features):\n", X[:5])  # Print first 5 rows of X
    print("First 5 values of y (label):\n", y[:5])   # Print first 5 values of y
 

    # Build, compile, and train the model
    model = build_model(X, layer_configurations,network_type,problem_type,num_classes=num_class)
    model, history = compile_and_train_model(model, X, y, problem_type, test_size=0.2, random_state=42, optimizer=optimizer, epochs=epochs, batch_size=32)

    # Save the model and history
    model_save_path, history_file = save_model_and_history(model, history, dataset_name)
    download_links = {
        "model": f"/download/{os.path.basename(model_save_path)}",
        "history": f"/download/{os.path.basename(history_file)}"
    }
    # Respond with success and paths to saved files
    return jsonify({'message': 'Model trained successfully!', 'model_path': model_save_path, 'history_path': history_file,'downloads': download_links})


@app.route('/MDSupPrjt/view_training_history/<file_name>')
def view_training_history(file_name):
    history_file = os.path.join('model_hist', f'history_{file_name}.csv')

    # Load the training history from CSV
    if os.path.exists(history_file):
        history_data = pd.read_csv(history_file)

        # Check if the necessary columns exist
        has_accuracy = 'accuracy' in history_data.columns
        has_val_accuracy = 'val_accuracy' in history_data.columns
        has_loss = 'loss' in history_data.columns
        has_val_loss = 'val_loss' in history_data.columns

        # Create a figure with two subplots: one for accuracy and one for loss
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Subplot 1: Accuracy
        if has_accuracy:
            ax[0].plot(history_data['accuracy'], label='Training Accuracy', color='blue')
        if has_val_accuracy:
            ax[0].plot(history_data['val_accuracy'], label='Validation Accuracy', color='green')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_title(f'Accuracy for {file_name}')
        ax[0].legend()

        # Subplot 2: Loss
        if has_loss:
            ax[1].plot(history_data['loss'], label='Training Loss', color='red')
        if has_val_loss:
            ax[1].plot(history_data['val_loss'], label='Validation Loss', color='orange')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].set_title(f'Loss for {file_name}')
        ax[1].legend()

        # Save the plot to a BytesIO object
        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)

        # Convert the image to base64 and pass it to the template
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('training_history.html', plot_url=plot_url)
    else:
        return f"History file for {file_name} not found."
@app.route('/Downloads/<path:filename>')
def download_file(filename):
    temp_folder = 'model_hist'  # Make sure this folder exists and contains the model files
    return send_from_directory(temp_folder, filename, as_attachment=True)

    
if __name__ == '__main__':
    app.run(debug=True)
