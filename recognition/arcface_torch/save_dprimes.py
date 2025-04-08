import os
import argparse

def save_text_file_contents(root_folder, output_file):
    # Open the output file in write mode
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through the root folder and its subdirectories
        for subdir, _, files in os.walk(root_folder):
            for file in files:
                # Process only .txt files
                if file.endswith('.txt'):
                    file_path = os.path.join(subdir, file)
                    try:
                        # Read the content of the text file
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                        # Write the filename and content to the output file
                        outfile.write(f"{file},")
                        outfile.write(f"{content}\n")
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

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
    
    # Call the function to process files
    save_text_file_contents(args.root_folder, args.output_file)
    print(f"All text files have been processed and saved in {args.output_file}.")

if __name__ == "__main__":
    main()

