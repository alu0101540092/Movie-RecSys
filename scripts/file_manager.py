import os
import glob
import sys
import shutil

# Default chunk size: 45MB (Github limit is 100MB, but 50MB is safe for non-LFS)
CHUNK_SIZE = 45 * 1024 * 1024

def split_file(filepath, chunk_size=CHUNK_SIZE):
    """Splits a file into multiple parts."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    file_size = os.path.getsize(filepath)
    if file_size <= chunk_size:
        print(f"Skipping {filepath}: smaller than chunk size.")
        return

    print(f"Splitting {filepath} ({file_size / (1024*1024):.2f} MB)...")
    
    with open(filepath, 'rb') as f:
        part_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            part_filename = f"{filepath}.part{part_num:03d}"
            with open(part_filename, 'wb') as part_file:
                part_file.write(chunk)
            
            print(f"Created {part_filename}")
            part_num += 1
            
    print(f"Done splitting {filepath}.")

def join_file(filepath):
    """Reconstructs a file from its parts."""
    # Check if file already exists
    if os.path.exists(filepath):
        # We could check md5 here to be sure, but existence is a good first check
        # For now, let's assume if it exists, it's good. 
        # But if the parts exist and are newer, maybe we should rebuild?
        # Let's keep it simple: if missing, rebuild.
        return

    parts = sorted(glob.glob(f"{filepath}.part*"))
    if not parts:
        # No parts found, cannot reconstruct
        return

    print(f"Reconstructing {filepath} from {len(parts)} parts...")
    
    try:
        with open(filepath, 'wb') as output_file:
            for part_path in parts:
                with open(part_path, 'rb') as part_file:
                    shutil.copyfileobj(part_file, output_file)
        print(f"Successfully reconstructed {filepath}")
    except Exception as e:
        print(f"Error reconstructing {filepath}: {e}")
        # Clean up partial file
        if os.path.exists(filepath):
            os.remove(filepath)

def process_directory(directory, action, extensions=None):
    """Recursively process a directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            
            # Skip existing parts
            if ".part" in file:
                continue
                
            # Filter by extension if provided
            if extensions and not any(file.endswith(ext) for ext in extensions):
                continue

            if action == 'split':
                if os.path.getsize(filepath) > CHUNK_SIZE:
                    split_file(filepath)
            elif action == 'join':
                # For join, we actually need to look for target files that DO NOT exist but have parts
                # This logic is a bit tricky with walk. 
                # Better to just look for .part000 files and deduce the target.
                pass

def join_all_in_directory(directory):
    """Finds all .part000 files and attempts to reconstruct their targets."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".part000"):
                # Target file is the part filename without the suffix
                target_path = os.path.join(root, file[:-8]) 
                join_file(target_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python file_manager.py [split-all|join-all|split <file>|join <file>]")
        sys.exit(1)

    command = sys.argv[1]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if command == "split-all":
        # Specific targets derived from observation
        targets = [
            os.path.join(base_dir, "datasets", "ml-32m", "ratings.csv"),
            os.path.join(base_dir, "datasets", "ml-32m", "tags.csv"),
        ]
        # And all .npy files in models
        models_dir = os.path.join(base_dir, "models")
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith(".npy"):
                    targets.append(os.path.join(models_dir, file))
                    
        for target in targets:
            split_file(target)
            
    elif command == "join-all":
        join_all_in_directory(base_dir)
        
    elif command == "split" and len(sys.argv) > 2:
        split_file(sys.argv[2])
        
    elif command == "join" and len(sys.argv) > 2:
        join_file(sys.argv[2])
