
import os
import shutil
import subprocess

# Paths
source_folder = "../CropSense-Face-Detection/output/face_cropped"
destination_folder = "../face_matcher/input"
compare_script = "compare_faces.py"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Allowed image extensions
image_extensions = (".jpg", ".jpeg", ".png")

# Step 1: Copy image files
if not os.path.exists(source_folder):
    print(f"[ERROR] Source folder not found: {source_folder}")
    exit()

for filename in os.listdir(source_folder):
    if filename.lower().endswith(image_extensions):
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)
        shutil.copy2(src_path, dst_path)
        print(f"✅ Copied: {filename}")

print("\n📦 All images copied successfully.")
print("🚀 Running face comparison script...\n")

# Step 2: Run compare_faces.py
try:
    result = subprocess.run(["python", "compare_faces.py"], capture_output=True, text=True)
    
    # Output from the script
    print("📄 Script Output:\n")
    print(result.stdout)

    # If there are errors
    if result.stderr:
        print("⚠️ Script Errors:\n")
        print(result.stderr)

except Exception as e:
    print(f"❌ Failed to run compare_faces.py: {e}")

