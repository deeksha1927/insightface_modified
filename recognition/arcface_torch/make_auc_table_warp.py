import os
import pandas as pd
import argparse

def read_text_files_to_table(folder_path):
    # Define the column names
    columns = [
        "Model Name" , "WPUT_warped_lr", "WPUT_warped", "opib_warped_lr", "opib_warped", "earvn_warped_lr", "EarVN1.0_warped" ,"awe_warped_lr", "awe_warped", "AWE-Ex_New_images_warped_lr", "AWE-Ex_New_images_warped"
    ]

    # Initialize an empty list to store the rows
    data = []

    # Iterate over all text files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            if "dprime" in file_name:
                continue
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
                    if "dprime" in key:
                        
                        continue
                    clean_key = key.replace("aucroc_", "").replace(".txt", "").strip()
                    print(clean_key)
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



