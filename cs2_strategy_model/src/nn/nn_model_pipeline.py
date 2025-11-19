# src/nn/player_zone_pipeline.py

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# CONFIG
# -------------------------------
TICKS_FOLDER = r"C:\gatech\ticks"
OUTPUT_FOLDER = "nn/output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

NUMERIC_FEATURES = [
    "X", "Y", "health", "teammates_alive", "enemies_alive",
    "total_rounds_played", "team_rounds_total"
]
CATEGORICAL_FEATURES = [
    "current_zone", "active_weapon_name", "teammate_1_zone", "teammate_2_zone",
    "teammate_3_zone", "teammate_4_zone", "demo_name"
]
TARGET_COL = "next_zone"
SEQ_LEN = 5  # sequence length for LSTM

# -------------------------------
# DATA LOADING
# -------------------------------
def load_and_combine_files(ticks_folder=TICKS_FOLDER):
    dfs = []
    for file in os.listdir(ticks_folder):
        if file.endswith(".csv") or file.endswith(".xlsx"):
            filepath = os.path.join(ticks_folder, file)
            match = re.search(r"-([^-\n]+)_ticks", file)
            map_name = match.group(1) if match else "unknown"

            if file.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            # Unique key for demo + player
            df["demo_name"] = df["demo_id"].astype(str) + "_" + df["name"].astype(str)
            df["map_name"] = map_name
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# -------------------------------
# PREPROCESSING
# -------------------------------
def preprocess_data(df):
    # Numeric features
    X_num = df[NUMERIC_FEATURES].values
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Categorical features
    cat_data = df[CATEGORICAL_FEATURES].astype(str)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(cat_data)

    # Target
    target_ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    y = target_ohe.fit_transform(df[[TARGET_COL]].astype(str))

    # Combine numeric + categorical
    X = np.concatenate([X_num_scaled, X_cat], axis=1)
    return X, y, ohe, target_ohe, scaler

# -------------------------------
# CREATE LSTM SEQUENCES
# -------------------------------
def create_lstm_sequences(X, y, seq_len=SEQ_LEN):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)

# -------------------------------
# MODEL TRAINING
# -------------------------------
def train_feedforward(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(128, activation="relu", input_dim=X_train.shape[1]),
        Dense(64, activation="relu"),
        Dense(y_train.shape[1], activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=50, batch_size=256, callbacks=[es], verbose=2
    )
    return model, history

def train_lstm(X_train, y_train, X_val, y_val):
    model = Sequential([
        LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dense(64, activation="relu"),
        Dense(y_train.shape[1], activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    es = EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=50, batch_size=256, callbacks=[es], verbose=2
    )
    return model, history

# -------------------------------
# EVALUATION
# -------------------------------
def one_hot_to_label(ohe, one_hot_array):
    return ohe.inverse_transform(one_hot_array)

def evaluate_model(model, X, y, target_ohe):
    y_pred_prob = model.predict(X)
    y_pred_labels = one_hot_to_label(target_ohe, y_pred_prob)
    y_true_labels = one_hot_to_label(target_ohe, y)
    print(classification_report(y_true_labels, y_pred_labels))
    return y_true_labels, y_pred_labels

def plot_confusion_matrix(y_true, y_pred, mapname, model_type):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Zone")
    plt.ylabel("True Zone")
    plt.title(f"Confusion Matrix - {mapname} ({model_type})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{mapname}_{model_type}_confusion.png"))
    plt.close()

# -------------------------------
# PLOTTING TRAINING METRICS
# -------------------------------
def plot_training_history(history, mapname, model_type):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_type} Loss - {mapname}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{model_type} Accuracy - {mapname}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_FOLDER, f"{mapname}_{model_type}_training.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plot saved at {plot_path}")

# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    df_all = load_and_combine_files()

    for mapname in df_all["map_name"].unique():
        print(f"\n--- Processing map: {mapname} ---")
        df_map = df_all[df_all["map_name"] == mapname].reset_index(drop=True)

        # Preprocess
        X, y, ohe, target_ohe, scaler = preprocess_data(df_map)

        # Split train/val for FFNN
        split_idx = int(0.8 * len(df_map))
        X_train_ff, X_val_ff = X[:split_idx], X[split_idx:]
        y_train_ff, y_val_ff = y[:split_idx], y[split_idx:]

        # Train Feedforward NN
        ff_model, ff_history = train_feedforward(X_train_ff, y_train_ff, X_val_ff, y_val_ff)
        ff_model_path = os.path.join(OUTPUT_FOLDER, f"{mapname}_feedforward.h5")
        ff_model.save(ff_model_path)
        print(f"Feedforward model saved at {ff_model_path}")

        y_true_ff, y_pred_ff = evaluate_model(ff_model, X_val_ff, y_val_ff, target_ohe)
        plot_confusion_matrix(y_true_ff, y_pred_ff, mapname, "FFNN")
        plot_training_history(ff_history, mapname, "FFNN")

        # Create LSTM sequences
        X_seq, y_seq = create_lstm_sequences(X, y)
        split_idx_seq = int(0.8 * len(X_seq))
        X_train_lstm, X_val_lstm = X_seq[:split_idx_seq], X_seq[split_idx_seq:]
        y_train_lstm, y_val_lstm = y_seq[:split_idx_seq], y_seq[split_idx_seq:]

        # Train LSTM
        lstm_model, lstm_history = train_lstm(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm)
        lstm_model_path = os.path.join(OUTPUT_FOLDER, f"{mapname}_lstm.h5")
        lstm_model.save(lstm_model_path)
        print(f"LSTM model saved at {lstm_model_path}")

        y_true_lstm, y_pred_lstm = evaluate_model(lstm_model, X_val_lstm, y_val_lstm, target_ohe)
        plot_confusion_matrix(y_true_lstm, y_pred_lstm, mapname, "LSTM")
        plot_training_history(lstm_history, mapname, "LSTM")

if __name__ == "__main__":
    main()
