import cv2
import numpy as np
import pandas as pd
import os

# ============ CONFIGURATION ============
# Paths
selected_faces_folder = r"C:\Yourpath\selected_faces"
model_prototxt = r"C:\Yourpath\age.prototxt.txt"  # UPDATE THIS PATH
model_weights = r"C:\Yourpath\dex_imdb_wiki.caffemodel"  

# CSV with metadata
metadata_csv = os.path.join(selected_faces_folder, 'faces_metadata.csv')

# ============ LOAD THE DEX MODEL ============
print("Loading DEX age prediction model...")
try:
    net = cv2.dnn.readNetFromCaffe(model_prototxt, model_weights)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure opencv-python is installed: pip install opencv-python")
    exit(1)

# ============ LOAD METADATA ============
df = pd.read_csv(metadata_csv)
print(f"\nLoaded metadata for {len(df)} faces")

# ============ PREDICT AGES ============
predicted_ages = []

for idx, row in df.iterrows():
    filename = row['filename']
    image_path = os.path.join(selected_faces_folder, filename)

    # Read image
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Could not read {filename}")
        predicted_ages.append(None)
        continue

    # Preprocess image for DEX model
    # DEX expects 224x224 BGR image
    blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224),
                                 (104, 117, 123),  # Mean subtraction (BGR)
                                 swapRB=False,  # Keep BGR format
                                 crop=False)

    # Set input and forward pass
    net.setInput(blob)
    predictions = net.forward()

    # DEX outputs a probability distribution over 101 age classes (0-100)
    # Expected age = weighted sum
    age_classes = np.arange(0, 101)
    predicted_age = np.sum(predictions[0] * age_classes)

    predicted_ages.append(predicted_age)

    true_age = row['true_age']
    age_diff = predicted_age - true_age

    print(f"✅ {filename}: True={true_age}, Predicted={predicted_age:.1f}, Diff={age_diff:+.1f}")

# ============ ADD PREDICTIONS TO DATAFRAME ============
df['predicted_age'] = predicted_ages
df['age_difference'] = df['predicted_age'] - df['true_age']

# Save updated CSV
output_csv = os.path.join(selected_faces_folder, 'faces_with_predictions.csv')
df.to_csv(output_csv, index=False)

print(f"\n{'=' * 60}")
print(f"✅ Predictions complete!")
print(f"📄 Results saved to: {output_csv}")
print(f"{'=' * 60}")

# ============ STATISTICS ============
print("\n📊 Prediction Statistics:")
print(f"Mean Absolute Error: {df['age_difference'].abs().mean():.2f} years")
print(f"Mean Error (bias): {df['age_difference'].mean():.2f} years")
print(f"Std Dev of Error: {df['age_difference'].std():.2f} years")

print("\nBy Gender:")
for gender in ['Male', 'Female']:
    gender_df = df[df['gender'] == gender]
    mae = gender_df['age_difference'].abs().mean()
    bias = gender_df['age_difference'].mean()
    print(f"  {gender}: MAE={mae:.2f}, Bias={bias:+.2f}")

print("\nAge prediction distribution:")
print(df['predicted_age'].describe())
