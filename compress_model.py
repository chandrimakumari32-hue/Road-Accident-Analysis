import joblib
import os

input_file = "severity_rf_model.pkl"
output_file = "severity_rf_model_compressed.pkl"

# Check original size
original_size = os.path.getsize(input_file) / (1024 ** 3)
print(f"Original size: {original_size:.2f} GiB")

print("Loading model...")

model = joblib.load(input_file)

print("Compressing model...")

joblib.dump(model, output_file, compress=3)

# Check compressed size
compressed_size = os.path.getsize(output_file) / (1024 ** 3)
print(f"Compressed size: {compressed_size:.2f} GiB")

if compressed_size < 2:
    print("Model is below GitHub LFS's 2 GiB file limit.")
else:
    print("Model is still too large for GitHub LFS.")