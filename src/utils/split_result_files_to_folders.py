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

        # 1. Determine the algorithm folder (check longer names first to avoid partial matches)
        if "DFBnB" in filename:
            algorithm = "DFBnB"
        elif "IDA" in filename:
            algorithm = "IDA"
        elif "XMM" in filename:
            algorithm = "XMM"
        elif "_A_" in filename:
            algorithm = "A"
        else:
            print(f"Skipping unrecognized algorithm: {filename}")
            continue

        # 2. Determine the domain category folder
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
            print(f"Skipping unrecognized domain: {filename}")
            continue

        # 3. Determine the lookahead folder
        lookahead_folder = None
        for i in range(1, 5):  # Checks 1 through 4
            if f"{i}lookahead" in filename:
                lookahead_folder = f"{i}_lookahead"
                break

        if not lookahead_folder:
            print(f"Skipping file with no lookahead value: {filename}")
            continue

        # 4. Construct target path: e.g., /your/path/DFBnB/LSP_Grids/1_lookahead/
        target_dir = base_path / algorithm / category / lookahead_folder

        # Create the directories if they don't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # 5. Move the file
        target_file_path = target_dir / filename
        shutil.move(str(file_path), str(target_file_path))
        print(f"Moved: {filename}  -->  {algorithm}/{category}/{lookahead_folder}/")

    print("\nFile organization complete!")

if __name__ == "__main__":
    # DIRECTORY PATH
    target_path = "/home/tzur-shubi/Documents/Programming/BiHS/results/2026_05_10" 
    
    organize_results(target_path)