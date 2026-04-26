import os
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
# Set this to the path where your results files are located
results_dir = "/home/tzur-shubi/Documents/Programming/BiHS/results/2026_04_26"

def organize_results(base_path):
    # Define the lookahead depths we want to search for
    lookaheads = [1, 2, 3, 4, 5, 6]

    # Iterate through everything in the specified directory
    for filename in os.listdir(base_path):
        
        # Construct the full path to the file
        filepath = os.path.join(base_path, filename)
        
        # Skip directories and Python scripts
        if not os.path.isfile(filepath) or filename.endswith('.py'):
            continue

        # 1. Determine the main folder (snake or LSP)
        if 'snake' in filename:
            main_folder = 'Snake_Grids'
        else:
            # Assumes anything without 'snake' in the name is LSP
            main_folder = 'LSP_Grids'

        # 2. Determine the lookahead subfolder
        target_subfolder = None
        for la in lookaheads:
            if f"{la}lookahead" in filename:
                target_subfolder = f"{la}la"
                break
        
        # 3. If a valid lookahead pattern was found, create the path and move the file
        if target_subfolder:
            # Construct the nested path inside the results directory
            target_dir = os.path.join(base_path, main_folder, target_subfolder)
            
            # Create the nested directories safely
            os.makedirs(target_dir, exist_ok=True)
            
            # Move the file into the target directory
            shutil.move(filepath, os.path.join(target_dir, filename))
            print(f"Moved: {filename}  -->  {os.path.join(main_folder, target_subfolder)}/")

if __name__ == "__main__":
    # Ensure the directory exists before trying to organize it
    if os.path.exists(results_dir):
        print(f"Organizing files in: {results_dir}...")
        organize_results(results_dir)
        print("Sorting complete!")
    else:
        print(f"Error: The directory '{results_dir}' does not exist. Please check the path in the CONFIGURATION block.")