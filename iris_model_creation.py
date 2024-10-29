import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import json
import pickle

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

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

# # Save metrics to a JSON file
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Labels', marker='o', linestyle='-', color='blue', markersize=6)
plt.plot(y_pred, label='Predictions', marker='x', linestyle='--', color='orange', markersize=8)

# Add a horizontal line for reference (optional)
plt.axhline(y=0.5, color='red', linestyle=':', label='Threshold')

# Title and labels
plt.title('True Labels vs Predictions', fontsize=16)
plt.xlabel('Sample Index', fontsize=14)
plt.ylabel('Class Label', fontsize=14)

# Customize x-ticks
plt.xticks(ticks=np.arange(0, len(y_test), step=max(1, len(y_test)//10)), 
           labels=np.arange(0, len(y_test), step=max(1, len(y_test)//10)))

# Add grid lines for better readability
plt.grid(True)

# Legend
plt.legend(fontsize=12)

# Save the plot
plt.savefig('static/predictions_vs_true.png')  # Save plot as image
plt.close()  # Close the plot to free up memory


# Save confusion matrix plot
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
plt.close()


# Generate classification report
class_report = classification_report(y_test, y_pred)

# Save classification report as an image
plt.figure(figsize=(10, 6))
plt.text(0.01, 1.25, class_report, {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')  # Turn off the axis
plt.title('Classification Report', fontsize=10)

# Save the classification report as an image
plt.savefig('static/classification_report.png', bbox_inches='tight', dpi=300)
plt.close()