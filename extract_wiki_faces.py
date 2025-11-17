import scipy.io
import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime

# ============ CONFIGURATION ============
# Path to wiki.mat file
wiki_mat_path = r"C:\Users\1650491\OneDrive - Universiteit Utrecht\Documents\PhD-2024\NWO-2024\HR design\codes\Algorithm\wiki_crop\wiki_crop\wiki.mat"

# Path to wiki images folder (where folders 00-99 are)
wiki_images_base = r"C:\Users\1650491\OneDrive - Universiteit Utrecht\Documents\PhD-2024\NWO-2024\HR design\codes\Algorithm\wiki_crop\wiki_crop"

# Where to save selected faces
output_folder = r"C:\Users\1650491\OneDrive - Universiteit Utrecht\Documents\PhD-2024\NWO-2024\HR design\codes\Algorithm\selected_faces"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# ============ LOAD WIKI METADATA ============
print("Loading WIKI metadata from .mat file...")
try:
    mat_data = scipy.io.loadmat(wiki_mat_path)
    print("✅ Metadata loaded successfully!")
except Exception as e:
    print(f"❌ Error loading .mat file: {e}")
    exit(1)

# Extract data from the .mat structure
wiki = mat_data['wiki'][0, 0]

# Get the arrays we need
dob = wiki['dob'][0]  # Date of birth (Matlab datenum format)
photo_taken = wiki['photo_taken'][0]  # Year photo was taken
full_path = wiki['full_path'][0]  # Paths to images
gender = wiki['gender'][0]  # Gender (1=male, 0=female, NaN=unknown)
name = wiki['name'][0]  # Person's name

print(f"Total entries in dataset: {len(dob)}")


# ============ CALCULATE AGES ============
def calculate_age(matlab_dob, photo_year):
    """
    Calculate age from Matlab datenum and photo year
    """
    if matlab_dob == 0 or photo_year == 0 or np.isnan(matlab_dob) or np.isnan(photo_year):
        return None

    # Convert Matlab datenum to year
    # Matlab datenum: days since January 0, 0000
    # Python datetime: days since January 1, 1970
    # Offset: 719529 days
    birth_datetime = datetime.fromordinal(int(matlab_dob)) + datetime.resolution * (matlab_dob % 1)
    birth_year = birth_datetime.year

    age = int(photo_year - birth_year)
    return age


# ============ COLLECT VALID SAMPLES ============
print("\nFiltering valid samples...")

valid_samples = []

for i in range(len(dob)):
    age = calculate_age(dob[i], photo_taken[i])

    # Filter criteria: age between 18-70, known gender
    if age is not None and 18 <= age <= 70:
        # Check if gender is valid (not NaN)
        gen = gender[i]
        if not np.isnan(gen):
            # Get image path
            img_relative_path = str(full_path[i][0])
            img_full_path = os.path.join(wiki_images_base, img_relative_path)

            # Check if image file exists
            if os.path.exists(img_full_path):
                person_name = str(name[i][0]) if len(name[i]) > 0 else "Unknown"

                valid_samples.append({
                    'path': img_full_path,
                    'relative_path': img_relative_path,
                    'age': age,
                    'gender': 'Male' if gen == 1 else 'Female',
                    'name': person_name,
                    'photo_year': int(photo_taken[i])
                })

print(f"✅ Found {len(valid_samples)} valid samples (age 18-70, known gender, existing files)")

if len(valid_samples) == 0:
    print("❌ No valid samples found! Check your paths.")
    exit(1)

# ============ SELECT 15 FEMALES AND 15 MALES ============
np.random.seed(42)  # For reproducibility

# Separate samples by gender
female_samples = [s for s in valid_samples if s['gender'] == 'Female']
male_samples = [s for s in valid_samples if s['gender'] == 'Male']

print(f"\nAvailable samples:")
print(f"  - Females: {len(female_samples)}")
print(f"  - Males: {len(male_samples)}")

# Check if we have enough samples
if len(female_samples) < 25:
    print(f"⚠️ Warning: Only {len(female_samples)} female samples available (need 15)")
if len(male_samples) < 25:
    print(f"⚠️ Warning: Only {len(male_samples)} male samples available (need 15)")

# Select 15 of each
num_females = min(25, len(female_samples))
num_males = min(25, len(male_samples))

selected_females = np.random.choice(female_samples, size=num_females, replace=False)
selected_males = np.random.choice(male_samples, size=num_males, replace=False)

# Combine them
selected_samples = list(selected_females) + list(selected_males)

print(f"\nSelected: {num_females} females + {num_males} males = {len(selected_samples)} total")

# ============ COPY IMAGES AND CREATE CSV ============
results = []

for idx, sample in enumerate(selected_samples, 1):
    # Create new filename
    new_filename = f"face_{idx:02d}.jpg"
    destination_path = os.path.join(output_folder, new_filename)

    # Copy image
    try:
        shutil.copy2(sample['path'], destination_path)

        results.append({
            'filename': new_filename,
            'true_age': sample['age'],
            'gender': sample['gender'],
            'name': sample['name'],
            'photo_year': sample['photo_year'],
            'original_path': sample['relative_path']
        })

        print(
            f"✅ [{idx:02d}/30] {new_filename} - Age: {sample['age']}, Gender: {sample['gender']}, Name: {sample['name']}")

    except Exception as e:
        print(f"❌ Error copying {sample['path']}: {e}")

# ============ SAVE METADATA TO CSV ============
if results:
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, 'faces_metadata.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"✅ SUCCESS! Extracted {len(results)} faces")
    print(f"📁 Images saved to: {output_folder}")
    print(f"📄 Metadata saved to: {csv_path}")
    print(f"{'=' * 60}")

    print("\nAge distribution:")
    print(df['true_age'].describe())

    print("\nGender distribution:")
    print(df['gender'].value_counts())
else:
    print("\n❌ No images were successfully copied")


