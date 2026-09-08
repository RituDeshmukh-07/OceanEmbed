import numpy as np
import matplotlib.pyplot as plt

from ml import config
from ml.preprocessing import load_norm_stats
from ml.predict import predict_from_patch


# Load one real held-out test sample
data = np.load(config.TEST_SAMPLES_PATH)
stats = load_norm_stats()

x = data["inputs"][0]

# Convert normalized SST/SSS back to physical units
input_mean = np.array(stats["input_mean"], dtype=np.float32).reshape(2, 1, 1)
input_std = np.array(stats["input_std"], dtype=np.float32).reshape(2, 1, 1)

x = x * input_std + input_mean

# Real model prediction
prediction = predict_from_patch(x)

depths = np.array(config.TARGET_DEPTHS)

# Print values
print("\nOceanEmbed Prediction")
print("=" * 40)

for depth, temp in zip(depths, prediction):
    print(f"{depth:>6} m : {temp:>7.2f} °C")

# Plot temperature profile
plt.figure(figsize=(7, 8))

plt.plot(prediction, depths, marker="o")

plt.gca().invert_yaxis()

plt.xlabel("Predicted Temperature (°C)")
plt.ylabel("Depth (m)")
plt.title("OceanEmbed - Predicted Temperature Profile")

plt.grid(alpha=0.3)
plt.tight_layout()

output_path = config.PLOTS_DIR / "prediction_profile.png"
plt.savefig(output_path, dpi=200)
plt.close()

print(f"\nSaved prediction plot -> {output_path}")