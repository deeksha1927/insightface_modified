import os
import pandas as pd
import argparse

def read_text_files_to_table(folder_path):
    # Define the column names
    columns = [
        "Model Name", "awe", "awe_lr", "awe_lr_rotated_cropped", "awe_rotated_crop",
        "AWE-Ex_New_images_rotated_crop", "AWE-Ex_New_images_rotated_crop_lr", "DIAST_lr",
        "EarVN1.0_rotated_crop", "EarVN1.0_rotated_crop_lr", "opib_lr", "opib_lr_rotated_crop", "WPUT_rotated_crop_lr", "WPUT_rotated_crop", "EarVN1.0_lr","AWE-Ex_New_images_lr" ,"WPUT_lr"
    ]

    # Initialize an empty list to store the rows
    data = []

    # Iterate over all text files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            model_name= os.path.splitext(file_name)[0]
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Ensure at least one line exists
            if not lines:
                continue

         

            # Create a dictionary to map column names to values
            row_data = {col: None for col in columns}
            row_data["Model Name"] = model_name

            # Map values from the file to corresponding column headings
            for line in lines[0:]:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    key, value = parts
                    # Clean the key to match the column name
                    clean_key = key.replace("dprime_", "").replace(".txt", "").strip()
                    if clean_key in columns:
                        row_data[clean_key] = float(value)

            # Append the row to the data list
            data.append([row_data[col] for col in columns])

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=columns)
    return df






def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Read text files in subfolders and save their filenames and contents.")
    parser.add_argument("root_folder", type=str, help="Path to the root folder containing text files.")
    parser.add_argument("output_file", type=str, help="Path to the output file where results will be saved.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Validate root folder
    if not os.path.isdir(args.root_folder):
        print(f"Error: The specified root folder does not exist: {args.root_folder}")
        return
    # Generate the table from the text files
    df = read_text_files_to_table(args.root_folder)
    
    df.to_csv(args.output_file, index=False)
    print(f"All text files have been processed and saved in {args.output_file}.")

if __name__ == "__main__":
    main()

