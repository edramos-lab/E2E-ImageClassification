import os

def scan_directory(root_dir):
    # Get the absolute path of the root directory
    root_dir = os.path.abspath(root_dir)
    
    with open('dataset_info.csv', 'w') as f:
        f.write("# Dataset Information\n")
        f.write("# Format: Full Path , Class Name\n")
        f.write("# --------------------------------\n\n")
        
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in sorted(os.listdir(class_dir)):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(class_dir, filename)
                        f.write(f"{full_path} , {class_name}\n")

if __name__ == "__main__":
    scan_directory('.') 