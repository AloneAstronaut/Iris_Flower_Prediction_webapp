import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Save the DataFrame to a CSV file
iris_df.to_csv('iris_dataset.csv', index=False)

print("Iris dataset saved as 'iris_dataset.csv'")
