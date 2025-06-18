
import os
import csv
import face_recognition
import cv2
from send_alert import send_alert  # Import the send_alert function

# Paths
database_path = "Database"
input_path = "input"
output_file = "match_results.csv"

# Step 1: Load and encode faces
print("\n📦 Encoding known faces...")
known_encodings = []
known_filenames = []

# Face encoding with upscaling and jittering for robustness
def load_and_encode(path, upscale_factor=2):
    img = face_recognition.load_image_file(path)
    img = cv2.resize(img, (0, 0), fx=upscale_factor, fy=upscale_factor)  # Upscale for low-res
    encodings = face_recognition.face_encodings(img, num_jitters=5)
    return encodings

# Encode database faces
for filename in os.listdir(database_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(database_path, filename)
        encodings = load_and_encode(path)
        if encodings:
            known_encodings.append(encodings[0])
            known_filenames.append(filename)
            print(f"✅ Encoded: {filename}")
        else:
            print(f"[SKIP] {filename} — No face found.")

# Step 2: Match input faces
print("\n🔍 Matching input faces...")
results = []

for filename in os.listdir(input_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(input_path, filename)
        encodings = load_and_encode(path)

        if not encodings:
            print(f"[{filename}] ❌ No face found.")
            results.append([filename, "No match", "—", "—"])
            continue

        matches = face_recognition.compare_faces(known_encodings, encodings[0], tolerance=0.3)
        if True in matches:
            idx = matches.index(True)
            matched_filename = known_filenames[idx]
            matched_name = os.path.splitext(matched_filename)[0]
            print(f"[{filename}] ✅ Match: {matched_filename}")
            
            # Send an email alert
            send_alert(matched_filename, matched_name, "navdeepnain49@gmail.com")  # Replace with actual email

            results.append([filename, matched_filename, matched_name, idx])
        else:
            print(f"[{filename}] ❌ No match.")
            results.append([filename, "No match", "—", "—"])

# Step 3: Save results to CSV
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Input Image", "Matched Image", "Matched Name", "Matched Index"])
    writer.writerows(results)

print(f"\n📝 Results saved to {output_file}")

