# ============================================================
# MODULE 3 — PREDICTION MODULE
# ============================================================
# Run module1_preprocessing.py and module2_lstm_gru_attention.py before this

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

Dense   = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Model   = tf.keras.models.Model

# ---------------------------
# Dense Layer (256)
# Learn complex nonlinear feature patterns
# ---------------------------
Dense_1 = Dense(256, activation='relu', name="Dense_Layer_1")(Flatten_Layer)

# ---------------------------
# Dropout
# Reduce overfitting during training
# ---------------------------
Drop_1 = Dropout(0.3)(Dense_1)

# ---------------------------
# Dense Layer (128)
# Further refine feature representation
# ---------------------------
Dense_2 = Dense(128, activation='relu', name="Dense_Layer_2")(Drop_1)

# ---------------------------
# Dropout
# Improve generalization
# ---------------------------
Drop_2 = Dropout(0.3)(Dense_2)

# ---------------------------
# Softmax Layer
# Compute probability distribution over crop classes
# ---------------------------
Output_Layer = Dense(
    num_classes,
    activation='softmax',
    name="Softmax_Output"
)(Drop_2)

# ---------------------------
# Model Creation
# ---------------------------
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

# ---------------------------
# Model Training
# ---------------------------
AACNet_Model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64
)

# ---------------------------
# Model Evaluation
# ---------------------------
loss, accuracy = AACNet_Model.evaluate(X_test, y_test)
print("\nTest Accuracy:", accuracy)

y_pred = np.argmax(AACNet_Model.predict(X_test), axis=1)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------
# Monte Carlo Dropout
# Estimate prediction uncertainty (T = 50 passes)
# ---------------------------
def mc_dropout_predict(model, data, T=50):
    predictions = []
    for _ in range(T):
        pred = model(data, training=True)   # dropout active at inference
        predictions.append(pred.numpy())
    predictions = np.array(predictions)    # shape: (T, 1, num_classes)
    mean_pred   = predictions.mean(axis=0)[0]
    std_pred    = predictions.std(axis=0)[0]
    return mean_pred, std_pred

# ---------------------------
# User Input
# ---------------------------
print("\nEnter Agricultural Parameters")
N    = float(input("Nitrogen     : "))
P    = float(input("Phosphorus   : "))
K    = float(input("Potassium    : "))
Temp = float(input("Temperature  : "))
Hum  = float(input("Humidity     : "))
ph   = float(input("pH value     : "))
Rain = float(input("Rainfall     : "))

user_data   = np.array([[N, P, K, Temp, Hum, ph, Rain]])
user_scaled = scaler.transform(user_data).reshape(1, 1, 7)

# ---------------------------
# Top-K Selection
# Select top 3 crops based on highest probabilities
# ---------------------------
mean_pred, std_pred = mc_dropout_predict(AACNet_Model, user_scaled, T=50)

top3_indices     = mean_pred.argsort()[::-1][:3]
top3_crops       = label_encoder.inverse_transform(top3_indices)
top3_confidences = mean_pred[top3_indices] * 100
top3_uncertainty = std_pred[top3_indices]  * 100

print("\n TOP 3 CROP RECOMMENDATIONS:\n")
for i in range(3):
    print(f"  {i+1}. {top3_crops[i]} : {top3_confidences[i]:.2f}% ± {top3_uncertainty[i]:.2f}%")

# ---------------------------
# Soil Fertility Output
# ---------------------------
fertility_score = calculate_soil_fertility(N, P, K, ph, Temp, Hum, Rain)
print(f"\n Soil Fertility Score: {fertility_score:.2f}/100")
if   fertility_score >= 75: print("Soil Status: Highly Fertile")
elif fertility_score >= 50: print("Soil Status: Moderately Fertile")
else:                        print("Soil Status: Low Fertility")
