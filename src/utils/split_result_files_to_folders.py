import shutil
from pathlib import Path

def organize_results(directory_path):
    # The variable with the directory path as requested
    base_path = Path(directory_path)
    
    # Ensure the base path exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: The directory '{base_path}' does not exist.")
        return

    # Iterate through all items in the directory
    for file_path in base_path.iterdir():
        # Skip if it's already a directory
        if not file_path.is_file():
            continue
            
        filename = file_path.name
        
        # 1. Determine the main category folder
        if "maze" in filename:
            category = "LSP_Mazes"
        elif "cube" in filename:
            category = "CIB"
        elif "grid" in filename:
            if "snake" in filename:
                category = "Snake_Grids"
            else:
                category = "LSP_Grids"
        else:
            # Skip files that don't match any known pattern
            print(f"Skipping unrecognized file: {filename}")
            continue
            
        # 2. Determine the lookahead folder
        lookahead_folder = None
        for i in range(1, 5): # Checks 1 through 4
            if f"{i}lookahead" in filename:
                lookahead_folder = f"{i}_lookahead"
                break
                
        if not lookahead_folder:
            print(f"Skipping file with no lookahead value: {filename}")
            continue
            
        # 3. Construct target path: e.g., /your/path/LSP_Grids/1_lookahead/
        target_dir = base_path / category / lookahead_folder
        
        # Create the directories if they don't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. Move the file
        target_file_path = target_dir / filename
        shutil.move(str(file_path), str(target_file_path))
        print(f"Moved: {filename}  -->  {category}/{lookahead_folder}/")

    print("\nFile organization complete!")

if __name__ == "__main__":
    # ---> CHANGE THIS VARIABLE TO YOUR ACTUAL DIRECTORY PATH <---
    target_path = "/home/tzur-shubi/Documents/Programming/BiHS/results/2026_05_05" 
    
    organize_results(target_path)