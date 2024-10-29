from flask import Flask, request, render_template, jsonify, url_for, redirect, session
import pickle
import numpy as np
import json
import os
from database import init_db, add_user, validate_user
from iris_model import train_and_save_model, model_exists

app = Flask(__name__)
app.secret_key = 'supersecretkey123'  # Set your secret key here

# Initialize the database here
init_db()

@app.route('/')
def main():
    return render_template('main.html')


@app.route('/index')
def index():
    model_trained = session.get('model_trained', False)  # Default to False if not set
    return render_template('index.html', model_trained=model_trained)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if validate_user(username, password):
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')


@app.route('/register', methods=['POST'])
def register():
    username = request.form['new_username']
    password = request.form['new_password']
    if add_user(username, password):
        return redirect(url_for('login'))
    else:
        return 'Username already exists', 400
    

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('model_trained', None)  # Clear model trained status on logout
    return redirect(url_for('main'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Train the model using the uploaded dataset
            train_and_save_model(file_path)  # Call the function to train the model

            # Update session to indicate model is trained
            session['model_trained'] = True
            
            return jsonify({"success": True, "message": "Model trained successfully."})
    return render_template('upload.html')

@app.route('/clear_model_status', methods=['POST'])
def clear_model_status():
    session.pop('model_trained', None)
    return jsonify(success=True)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        model = load_model()  # Load the trained model
        if model is None:
            return "Model not trained yet!", 400
        
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)

        species = ['Setosa', 'Versicolor', 'Virginica']
        return render_template('result.html', prediction=species[prediction[0]])
    
    return render_template('predict.html')  # Render the prediction form if GET request


@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    # Load metrics from JSON (ensure to load metrics correctly)
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    accuracy_percentage = metrics['accuracy'] * 100  # Convert to percentage
    return render_template('accuracy.html', accuracy=accuracy_percentage)


@app.route('/confusion_matrix', methods=['GET'])
def get_confusion_matrix():
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    conf_matrix = metrics['confusion_matrix']
    return render_template('confusion_matrix.html', confusion_matrix=conf_matrix)


@app.route('/classification_report', methods=['GET'])
def get_classification_report():
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    class_report = metrics['classification_report']
    return render_template('classification_report.html', report=class_report)


@app.route('/plot', methods=['GET'])
def plot():
    return render_template('plot.html', plot_url='plot')


def load_model():
    if os.path.exists('iris_model2.pkl'):
        with open('iris_model2.pkl', 'rb') as f:
            return pickle.load(f)
    return None


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
