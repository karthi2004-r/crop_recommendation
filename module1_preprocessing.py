# ============================================================
# MODULE 1 — DATA PREPROCESSING
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------------------
# Data Loading
# ---------------------------
data = pd.read_csv("Crop_recommendation.csv")

# ---------------------------
# Feature Separation
# ---------------------------
X = data.drop("label", axis=1)
y = data["label"]

# ---------------------------
# Min-Max Normalization
# ---------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Label Encoding
# ---------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ---------------------------
# Soil Fertility Score Computation
# ---------------------------
def calculate_soil_fertility(N, P, K, ph, Temp, Hum, Rain):
    n_score    = min(N / 140, 1)
    p_score    = min(P / 145, 1)
    k_score    = min(K / 205, 1)
    ph_score   = 1 - abs(ph - 7) / 7
    temp_score = 1 - abs(Temp - 25) / 25
    hum_score  = Hum / 100
    rain_score = Rain / 300
    fertility  = (
        0.25 * n_score +
        0.20 * p_score +
        0.20 * k_score +
        0.10 * ph_score +
        0.10 * temp_score +
        0.10 * hum_score +
        0.05 * rain_score
    )
    return fertility * 100

# ---------------------------
# Reshaping for RNN (Samples, Time Steps, Features)
# ---------------------------
X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# ---------------------------
# Train-Test Split (80/20)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded,
    test_size=0.2,
    random_state=42
)

num_classes = len(np.unique(y_encoded))

print("Module 1 complete.")
print(f"X_train shape : {X_train.shape}")
print(f"X_test  shape : {X_test.shape}")
print(f"Classes       : {num_classes}")
