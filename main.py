# ============================================================
# MAIN RUNNER — AACNet Crop Recommendation System
# Runs all 4 modules in sequence
# ============================================================
# HOW TO RUN:
#   python main.py
# ============================================================

# ============================================================
# MODULE 1 — DATA PREPROCESSING
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("=" * 50)
print(" MODULE 1 — Data Preprocessing")
print("=" * 50)

# Data Loading
data = pd.read_csv("Crop_recommendation.csv")

# Feature Separation
X = data.drop("label", axis=1)
y = data["label"]

# Min-Max Normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Label Encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Soil Fertility Score Computation
def calculate_soil_fertility(N, P, K, ph, Temp, Hum, Rain):
    n_score    = min(N / 140, 1)
    p_score    = min(P / 145, 1)
    k_score    = min(K / 205, 1)
    ph_score   = 1 - abs(ph - 7) / 7
    temp_score = 1 - abs(Temp - 25) / 25
    hum_score  = Hum / 100
    rain_score = Rain / 300
    fertility  = (
        0.25 * n_score   +
        0.20 * p_score   +
        0.20 * k_score   +
        0.10 * ph_score  +
        0.10 * temp_score +
        0.10 * hum_score  +
        0.05 * rain_score
    )
    return fertility * 100

# Reshaping for RNN (Samples, Time Steps, Features)
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42
)

num_classes = len(np.unique(y_encoded))

print(f"  X_train shape : {X_train.shape}")
print(f"  X_test  shape : {X_test.shape}")
print(f"  Classes       : {num_classes}")
print("  Module 1 DONE\n")


# ============================================================
# MODULE 2 — LSTM-GRU ATTENTION MODULE
# ============================================================

import tensorflow as tf 

Input       = tf.keras.layers.Input
LSTM        = tf.keras.layers.LSTM
GRU         = tf.keras.layers.GRU
Attention   = tf.keras.layers.Attention
Concatenate = tf.keras.layers.Concatenate
Flatten     = tf.keras.layers.Flatten
Dense       = tf.keras.layers.Dense
Dropout     = tf.keras.layers.Dropout
Model       = tf.keras.models.Model

print("=" * 50)
print(" MODULE 2 — LSTM-GRU Attention Architecture")
print("=" * 50)

# Input Layer
Input_Layer = Input(shape=(1, 7), name="Input_Features")

# LSTM Block — short-term nutrient & rainfall variations
LSTM_Block = LSTM(
    units=128,
    return_sequences=True,
    name="LSTM_Temporal_Learning"
)(Input_Layer)

# Attention Level 1 — Context Vector C1
Attention_1 = Attention(name="Attention_Level_1")(
    [LSTM_Block, LSTM_Block]
)

# GRU Block — long-term soil fertility impact
GRU_Block = GRU(
    units=128,
    return_sequences=True,
    name="GRU_Long_Term_Learning"
)(Attention_1)

# Attention Level 2 — Context Vector C2
Attention_2 = Attention(name="Attention_Level_2")(
    [GRU_Block, GRU_Block]
)

# Concatenation — merge C1 and C2 into feature vector F
Fusion        = Concatenate(name="Feature_Fusion")([Attention_1, Attention_2])
Flatten_Layer = Flatten(name="Flatten")(Fusion)

print("  Module 2 DONE\n")


# ============================================================
# MODULE 3 — PREDICTION MODULE
# ============================================================

from sklearn.metrics import classification_report

print("=" * 50)
print(" MODULE 3 — Prediction")
print("=" * 50)

# Dense Layer (256) — learn complex nonlinear feature patterns
Dense_1 = Dense(256, activation='relu', name="Dense_Layer_1")(Flatten_Layer)

# Dropout — reduce overfitting
Drop_1 = Dropout(0.3)(Dense_1)

# Dense Layer (128) — further refine feature representation
Dense_2 = Dense(128, activation='relu', name="Dense_Layer_2")(Drop_1)

# Dropout — improve generalization
Drop_2 = Dropout(0.3)(Dense_2)

# Softmax Layer — probability distribution over crop classes
Output_Layer = Dense(
    num_classes,
    activation='softmax',
    name="Softmax_Output"
)(Drop_2)

# Model Creation
AACNet_Model = Model(
    inputs=Input_Layer,
    outputs=Output_Layer,
    name="AACNet_Model"
)

AACNet_Model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

AACNet_Model.summary()

# Model Training
AACNet_Model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64
)

# Model Evaluation
loss, accuracy = AACNet_Model.evaluate(X_test, y_test)
print(f"\n  Test Accuracy : {accuracy:.4f}")

y_pred = np.argmax(AACNet_Model.predict(X_test), axis=1)
print("\n  Classification Report:\n")
print(classification_report(y_test, y_pred))

# Monte Carlo Dropout — T = 50 passes
def mc_dropout_predict(model, data, T=50):
    predictions = []
    for _ in range(T):
        pred = model(data, training=True)
        predictions.append(pred.numpy())
    predictions = np.array(predictions)
    mean_pred   = predictions.mean(axis=0)[0]
    std_pred    = predictions.std(axis=0)[0]
    return mean_pred, std_pred

# User Input
print("\n  Enter Agricultural Parameters")
N    = float(input("  Nitrogen     : "))
P    = float(input("  Phosphorus   : "))
K    = float(input("  Potassium    : "))
Temp = float(input("  Temperature  : "))
Hum  = float(input("  Humidity     : "))
ph   = float(input("  pH value     : "))
Rain = float(input("  Rainfall     : "))

user_data   = np.array([[N, P, K, Temp, Hum, ph, Rain]])
user_scaled = scaler.transform(user_data).reshape(1, 1, 7)

# Top-K Selection — top 3 crops
mean_pred, std_pred = mc_dropout_predict(AACNet_Model, user_scaled, T=50)

top3_indices     = mean_pred.argsort()[::-1][:3]
top3_crops       = label_encoder.inverse_transform(top3_indices)
top3_confidences = mean_pred[top3_indices] * 100
top3_uncertainty = std_pred[top3_indices]  * 100

print("\n  TOP 3 CROP RECOMMENDATIONS:\n")
for i in range(3):
    print(f"    {i+1}. {top3_crops[i]} : {top3_confidences[i]:.2f}% +/- {top3_uncertainty[i]:.2f}%")

# Soil Fertility Output
fertility_score = calculate_soil_fertility(N, P, K, ph, Temp, Hum, Rain)
print(f"\n  Soil Fertility Score : {fertility_score:.2f}/100")
if   fertility_score >= 75: print("  Soil Status : Highly Fertile")
elif fertility_score >= 50: print("  Soil Status : Moderately Fertile")
else:                        print("  Soil Status : Low Fertility")

print("  Module 3 DONE\n")


# ============================================================
# MODULE 4 — DECISION SUPPORT & EXPLAINABILITY
# ============================================================

import shap
import matplotlib.pyplot as plt

print("=" * 50)
print(" MODULE 4 — Explainability (SHAP)")
print("=" * 50)

feature_names = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Temperature", "Humidity", "pH", "Rainfall"
]

# Data Preparation for SHAP
X_train_flat = X_train.reshape(X_train.shape[0], 7)
user_flat    = user_scaled.reshape(1, 7)

# SHAP Explainer
def model_predict(data):
    data = data.reshape(data.shape[0], 1, 7)
    return AACNet_Model.predict(data)

explainer   = shap.KernelExplainer(model_predict, X_train_flat[:100])
shap_values = explainer.shap_values(user_flat)

# SHAP Contribution Extraction
shap_array      = np.array(shap_values)
predicted_index = top3_indices[0]

if shap_array.ndim == 3:
    contributions = shap_array[0, :, predicted_index]
else:
    contributions = shap_array.reshape(-1)[:7]

contributions   = contributions[:7]
sorted_indices  = np.argsort(np.abs(contributions))[::-1]
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_values   = contributions[sorted_indices]

# Feature Importance Bar Chart
plt.figure()
plt.barh(sorted_features, sorted_values)
plt.axvline(x=0)
plt.xlabel("SHAP Impact Value")
plt.title(f"Feature Impact on {top3_crops[0]} Recommendation")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("prediction_explanation.png")
plt.show()
print("  Saved: prediction_explanation.png")

# Top 3 Crop Probability Bar Chart
plt.figure()
plt.bar(top3_crops, top3_confidences)
plt.ylabel("Probability (%)")
plt.title("Top 3 Crop Recommendations")
plt.tight_layout()
plt.savefig("top3_crop_probability.png")
plt.show()
print("  Saved: top3_crop_probability.png")

# Human-Readable Explanation
impact_labels = [
    "Strong Positive Impact",
    "Positive Impact",
    "Favorable Condition"
]

print(f"\n  Recommended Crop  : {top3_crops[0]}")
print(f"  Confidence        : {np.max(mean_pred)*100:.0f}%")
print(f"\n  Explanation:")
print(f"  {top3_crops[0]} is recommended mainly because of:\n")

for idx, impact in zip(sorted_indices[:3], impact_labels):
    direction = "High" if contributions[idx] > 0 else "Low"
    print(f"    {direction} {feature_names[idx]} ({impact})")

print("\n  Module 4 DONE")
print("=" * 50)
print(" ALL MODULES COMPLETE")
print("=" * 50)
