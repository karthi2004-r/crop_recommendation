# ============================================================
# MODULE 4 — DECISION SUPPORT & EXPLAINABILITY MODULE
# ============================================================
# Run modules 1, 2, 3 before this module

import shap
import numpy as np
import matplotlib.pyplot as plt

feature_names = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Temperature", "Humidity", "pH", "Rainfall"
]

# ---------------------------
# Data Preparation for SHAP
# ---------------------------
X_train_flat = X_train.reshape(X_train.shape[0], 7)
user_flat    = user_scaled.reshape(1, 7)

# ---------------------------
# SHAP Explainer
# Compute global feature importance scores
# ---------------------------
def model_predict(data):
    data = data.reshape(data.shape[0], 1, 7)
    return AACNet_Model.predict(data)

explainer   = shap.KernelExplainer(model_predict, X_train_flat[:100])
shap_values = explainer.shap_values(user_flat)

# ---------------------------
# SHAP Contribution Extraction
# ---------------------------
shap_array      = np.array(shap_values)
predicted_index = top3_indices[0]

if shap_array.ndim == 3:
    contributions = shap_array[0, :, predicted_index]
else:
    contributions = shap_array.reshape(-1)[:7]

contributions  = contributions[:7]
sorted_indices = np.argsort(np.abs(contributions))[::-1]

sorted_features = [feature_names[i] for i in sorted_indices]
sorted_values   = contributions[sorted_indices]

# ---------------------------
# Feature Importance Visualization
# Bar chart ranking features by SHAP contribution
# ---------------------------
plt.figure()
plt.barh(sorted_features, sorted_values)
plt.axvline(x=0)
plt.xlabel("SHAP Impact Value")
plt.title(f"Feature Impact on {top3_crops[0]} Recommendation")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("prediction_explanation.png")
plt.show()
print("Explanation graph saved as 'prediction_explanation.png'")

# ---------------------------
# Top 3 Crop Probability Bar Chart
# ---------------------------
plt.figure()
plt.bar(top3_crops, top3_confidences)
plt.ylabel("Probability (%)")
plt.title("Top 3 Crop Recommendations")
plt.tight_layout()
plt.savefig("top3_crop_probability.png")
plt.show()
print("Top 3 crop probability graph saved as 'top3_crop_probability.png'")

# ---------------------------
# Human-Readable Explanation
# Identify dominant feature and generate natural language reason
# ---------------------------
impact_labels = [
    "Strong Positive Impact",
    "Positive Impact",
    "Favorable Condition"
]

print(f"\n Recommended Crop : {top3_crops[0]}")
print(f" Confidence        : {np.max(mean_pred)*100:.0f}%")
print(f"\n Explanation:")
print(f" {top3_crops[0]} is recommended mainly because of:\n")

for idx, impact in zip(sorted_indices[:3], impact_labels):
    direction = "High" if contributions[idx] > 0 else "Low"
    print(f"   {direction} {feature_names[idx]} ({impact})")
