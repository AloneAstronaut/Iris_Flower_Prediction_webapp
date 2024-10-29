import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json
import pickle
import os

def train_and_save_model(csv_file):
    # Load the dataset from CSV
    iris_df = pd.read_csv(csv_file)

    # Separate features and target
    X = iris_df.iloc[:, :-1].values  # All rows, all columns except the last one
    y = iris_df.iloc[:, -1].values     # All rows, only the last column

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # Save the model
    with open('iris_model2.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Generate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),  # Convert to list for JSON serialization
        'classification_report': class_report
    }

    # Save metrics to a JSON file
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Optionally, create plots
    create_plots(y_test, y_pred, conf_matrix)

def create_plots(y_test, y_pred, conf_matrix):
    # Create the plot for True Labels vs Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Labels', marker='o', linestyle='-', color='blue', markersize=6)
    plt.plot(y_pred, label='Predictions', marker='x', linestyle='--', color='orange', markersize=8)
    plt.axhline(y=0.5, color='red', linestyle=':', label='Threshold')
    plt.title('True Labels vs Predictions', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Class Label', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.savefig('static/predictions_vs_true.png')
    plt.close()

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(rotation=45)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png')
    plt.close()

    # Generate and save classification report as an image
    report = classification_report(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    plt.text(0.01, 1.25, report, {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.title('Classification Report', fontsize=10)
    plt.savefig('static/classification_report.png', bbox_inches='tight', dpi=300)
    plt.close()

def model_exists():
    """ Check if a model has already been trained and saved. """
    return os.path.exists('iris_model2.pkl')

if __name__ == "__main__":
    train_and_save_model('path_to_your_csv.csv')