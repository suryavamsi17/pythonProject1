import numpy as np
import pandas as pd

# Create a synthetic dataset
np.random.seed(42)
num_samples = 300
features = ['Feature1', 'Feature2']  # Replace with actual features

# Generate random data
data = {
    'CustomerID': np.arange(1, num_samples+1),
    'Feature1': np.random.randint(10, 100, num_samples),
    'Feature2': np.random.randint(20, 150, num_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
csv_filename = 'data.csv'
df.to_csv(csv_filename, index=False)

print(f"Dataset saved to {csv_filename}")
