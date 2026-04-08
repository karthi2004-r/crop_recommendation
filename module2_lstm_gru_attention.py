# ============================================================
# MODULE 2 — LSTM-GRU ATTENTION MODULE (AACNet Architecture)
# ============================================================
# Run module1_preprocessing.py before this module

import tensorflow as tf

Input       = tf.keras.layers.Input
LSTM        = tf.keras.layers.LSTM
GRU         = tf.keras.layers.GRU
Attention   = tf.keras.layers.Attention
Concatenate = tf.keras.layers.Concatenate
Flatten     = tf.keras.layers.Flatten
Model       = tf.keras.models.Model

# ---------------------------
# Input Layer
# ---------------------------
Input_Layer = Input(shape=(1, 7), name="Input_Features")

# ---------------------------
# LSTM Block
# Learns short-term nutrient and rainfall variations
# ---------------------------
LSTM_Block = LSTM(
    units=128,
    return_sequences=True,
    name="LSTM_Temporal_Learning"
)(Input_Layer)

# ---------------------------
# Attention Level 1
# Score → Weight → Context Vector C₁
# ---------------------------
Attention_1 = Attention(name="Attention_Level_1")(
    [LSTM_Block, LSTM_Block]
)

# ---------------------------
# GRU Block
# Learns long-term soil fertility impact
# ---------------------------
GRU_Block = GRU(
    units=128,
    return_sequences=True,
    name="GRU_Long_Term_Learning"
)(Attention_1)

# ---------------------------
# Attention Level 2
# Score → Weight → Context Vector C₂
# ---------------------------
Attention_2 = Attention(name="Attention_Level_2")(
    [GRU_Block, GRU_Block]
)

# ---------------------------
# Concatenation Layer
# Merge C₁ and C₂ into unified feature vector F
# ---------------------------
Fusion        = Concatenate(name="Feature_Fusion")([Attention_1, Attention_2])
Flatten_Layer = Flatten(name="Flatten")(Fusion)

print("Module 2 complete.")
print("Flatten_Layer ready for Module 3.")
