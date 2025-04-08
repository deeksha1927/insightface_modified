import os
import pandas as pd
import numpy as np

# Folder path containing the CSV files
folder_path = "/store01/flynn/darun/plots_ear/vit_l/"


# Initialize an empty DataFrame to hold all the data
all_data = pd.DataFrame()

# Iterate through all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Calculate the mean and std for each column (excluding the Model Name column)
        means = df.iloc[:, 1:].mean(axis=0)
        stds = df.iloc[:, 1:].std(axis=0)
        
        # Format the mean and std as "mean ± std"
        formatted_columns = [
            f"{mean:.4f} ± {std:.4f}" for mean, std in zip(means, stds)
        ]
        
        # Extract the model name from the file name (remove ".csv" extension)
        model_name = file_name.replace(".csv", "")
        
        # Create a new row with the dynamically derived model name
        new_row = pd.DataFrame([[model_name] + formatted_columns],
                               columns=['Model Name'] + df.columns[1:].tolist())
        
        # Append this row to all_data DataFrame
        all_data = pd.concat([all_data, new_row], ignore_index=True)

# Save the resulting DataFrame to a new CSV file
all_data.to_csv("/store01/flynn/darun/plots_ear/vit_l/aggregated_data.csv", index=False)
