import pandas as pd
import os

# Path to your CSV
selected_faces_folder = r"C:\Users\1650491\OneDrive - Universiteit Utrecht\Documents\PhD-2024\NWO-2024\HR design\codes\Algorithm\selected_faces"
csv_path = os.path.join(selected_faces_folder, 'faces_with_predictions.csv')

# Read the CSV
df = pd.read_csv(csv_path, decimal=',')  # Use comma if that's what's in your file

# Display the data nicely
print("=" * 80)
print("FACES WITH AGE PREDICTIONS")
print("=" * 80)
print(df.to_string())  # Shows full dataframe

# Or just show specific columns
print("\n" + "=" * 80)
print("AGE PREDICTIONS SUMMARY")
print("=" * 80)
print(df[['filename', 'gender', 'true_age', 'predicted_age', 'age_difference']])

# Show statistics
print("\n" + "=" * 80)
print("STATISTICS")
print("=" * 80)
print(f"Mean Absolute Error: {df['age_difference'].abs().mean():.2f} years")
print(f"Mean Bias: {df['age_difference'].mean():.2f} years")

# Check specific rows
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)
for idx, row in df.head(5).iterrows():
    print(f"{row['filename']}: True={row['true_age']}, Predicted={row['predicted_age']:.1f}, Diff={row['age_difference']:+.1f}")