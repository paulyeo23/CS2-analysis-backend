import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


local_ticks_folder_path = r"..\..\data"   # folder with lightweight replays for multiple maps so that the demo runs
nn_op_folder = "nn/output"
#os.makedirs(nn_op_folder, exist_ok=True)

#all of my numeric featurex
NUMERIC_FEATURES = ["X", "Y", "health", "teammates_alive", "enemies_alive","total_rounds_played", "team_rounds_total"]
# categorical features on input
CATEGORICAL_FEATURES = ["current_zone", "active_weapon_name","teammate_1_zone", "teammate_2_zone", "teammate_3_zone", "teammate_4_zone"]
# the lapel
TARGET_COL = "next_zone"
# model hyperparams
SEQ_LEN = 5
BATCH_SIZE = 128
EPOCHS = 5
VALIDATION_SPLIT = 0.2
PATIENCE = 5

def load_and_combine_files(ticks_folder=local_ticks_folder_path):
    dfs = []
    for file in os.listdir(ticks_folder):
        #print(f"got file {file}")
        if file.endswith(".csv") or file.endswith(".xlsx"):
            filepath = os.path.join(ticks_folder, file)
            match = re.search(r"-([^-\n]+)_ticks", file)
            map_name = match.group(1) if match else "unknown"
            #print(f"Loading {map_name} file: {filepath}")
            if file.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            # suspecting that the model needs to keep the player and demo movements toegehter
            # should not see player 1 move to player 2 posn so grouping them bu demo+player
            df["demo_name"] = df["demo_id"].astype(str) + "_" + df["name"].astype(str)
            df["map_name"] = map_name
            dfs.append(df)

    if len(dfs) == 0:
        print(f"No tick files found in folder:{ticks_folder}")
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    print(f"all data i have combined rows {len(combined)}")
    return combined

def preprocess_map(df_map):
    df = df_map.copy()
    print(f"Preprocessing map:{df['map_name'].iloc[0]} rows:{len(df)}")
    #print(df["is_alive"])
    #i am lookgin at players spectation also
    #shoudl not use those
    # is_alive might be boolean or strings; normalize truthiness
    df["is_alive_norm"] = df["is_alive"].astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])
    df = df[df["is_alive_norm"]].copy()
    #print("Rows after is_alive filter:", len(df))
    df = df.reset_index(drop=True)
    # fillna for num features
    X_num = df[NUMERIC_FEATURES].fillna(0).astype(float).values
    #scaling was a suggesting for spatial data so that maps done look squished
    #using standard scvalar
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    # current_zone keep raw strings to map to ints
    current_zone = df["current_zone"].astype(str).values
    # filter out rows where target is missing so that we dont have unknown predections
    before = len(df)
    df = df[~df[TARGET_COL].isna()].reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before-after} rows that had missing {TARGET_COL}")
    # need to recompute arrays to reflect dropped rows
    X_num_scaled = X_num_scaled[df.index.values]
    current_zone = current_zone[df.index.values]
    labels = df[TARGET_COL].astype(str).values

    return df, X_num_scaled, current_zone, labels, scaler


# Convert zones -> small integer indices
def build_zone_index_mapping(current_zone_array, label_array):
    # the zones could be in targer and current zone
    unique_zones = sorted(set(list(current_zone_array) + list(label_array)))
    zone_to_idx = {z:i for i,z in enumerate(unique_zones)}
    idx_to_zone = {i:z for z,i in zone_to_idx.items()}
    #print(f"Found zones: {unique_zones}")
    print("Found zones:", len(unique_zones))
    return zone_to_idx, idx_to_zone

# here we are building a sequence of steps for lstm moddel to see and learn
def create_lstm_sequences_per_player(df, X_num_scaled, current_zone, labels, zone_to_idx, seq_len=SEQ_LEN):
    X_seqs = []
    y_idxs = []
    change_flags = []
    num_feats = X_num_scaled.shape[1]
    feat_dim = num_feats + 1
    #print(f"features are {num_feats} and dim {feat_dim}")
    #add 1 for current_zone_idx
    grouped = df.groupby("demo_name", sort=False)
    print(f"Building sequences from {len(grouped)} player timelines")
    for demo_key, grp in grouped:
        grp = grp.sort_values("tick").reset_index(drop=True)
        if len(grp) <= seq_len:
            continue
        numeric_block = X_num_scaled[grp.index.values]
        zones_block = current_zone[grp.index.values]
        labels_block = labels[grp.index.values]
        zone_idx_block = np.array([zone_to_idx.get(z, -1) for z in zones_block], dtype=int)

        for i in range(len(grp) - seq_len):
            #this is the input window to tbe looked at
            X_window_num = numeric_block[i:i+seq_len]
            X_window = np.zeros((seq_len, feat_dim), dtype=float)
            X_window[:, :num_feats] = X_window_num
            X_window[:, num_feats] = zone_idx_block[i:i+seq_len]
            #print(f"winsow is {X_window_num}, window is {X_window}")
            #here are my target labels
            label_zone = labels_block[i + seq_len]
            label_idx = zone_to_idx.get(label_zone, -1)
            #print(f"labels are {label_zone}, {label_idx}")
            if label_idx < 0:
                continue

            X_seqs.append(X_window)
            y_idxs.append(label_idx)
            #print(f"xseq and yids  {X_seqs}, {y_idxs}")
            #need to track if the zone changed
            #created the change flags for knowing if zone chnaged
            #need this as most of the time the player stays at same place
            #print(f"******{zone_idx_block}")
            #print(f"******{i}, {seq_len}")
            current_idx = zone_idx_block[i + seq_len - 1]
            zone_changed = (current_idx != label_idx)
            change_flags.append(zone_changed)
            #print(f"********check change flasg {change_flags}")
    if len(X_seqs) == 0:
        #print(f"No seq created")
        return np.zeros((0, seq_len, feat_dim)), np.array([]), np.array([])

    return np.array(X_seqs), np.array(y_idxs), np.array(change_flags)

#build the model
#using mode common params for the model
def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(32, input_shape=(input_shape[0], input_shape[1]), return_sequences=False))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

#plot
def plot_training_history(history, mapname, out_prefix):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title("Accuracy")
    plt.suptitle(f"{mapname} LSTM Training")
    path = os.path.join(out_prefix, f"{mapname}_lstm_training.png")
    plt.savefig(path)
    plt.close()
    print("Saved training plot to:", path)

def plot_confusion_matrix(y_true_idx, y_pred_idx, idx_to_zone, mapname, out_prefix):
    labels = [idx_to_zone[i] for i in sorted(idx_to_zone.keys())]
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=sorted(idx_to_zone.keys()))
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=False, cmap="plasma", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{mapname} Confusion Matrix",pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    path = os.path.join(out_prefix, f"{mapname}_confusion.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print("Saved confusion matrix to:", path)


def load_and_preprocess_new_replay(new_replay_path, seq_len=SEQ_LEN):
    #adding this as the model should be tested on an unseen replay
    #print(f"Loading new replay file: {new_replay_path}")
    if new_replay_path.endswith(".csv"):
        df_new_replay = pd.read_csv(new_replay_path)
    else:
        df_new_replay = pd.read_excel(new_replay_path)
    match = re.search(r"-([^-\n]+)_ticks", new_replay_path)
    map_name = match.group(1) if match else "unknown"
    #print(f"******{df_new_replay}")
    #need mapname in df
    df_new_replay["map_name"] = map_name
    #also need demo name
    if "demo_id" in df_new_replay.columns and "name" in df_new_replay.columns:
        df_new_replay["demo_name"] = df_new_replay["demo_id"].astype(str) + "_" + df_new_replay["name"].astype(str)

    #follow same process as done for the training data
    df_new_replay_clean, X_num_scaled, current_zone, labels, scaler = preprocess_map(df_new_replay)
    #some file had a problem

    #building the zone inddex like training
    zone_to_idx, idx_to_zone = build_zone_index_mapping(current_zone, labels)
    #creating sequences like training
    X_seq, y_idx, _ = create_lstm_sequences_per_player(df_new_replay_clean, X_num_scaled, current_zone, labels, zone_to_idx, seq_len=seq_len)
    #covert the y to categories
    num_classes = len(zone_to_idx)
    y_cat = to_categorical(y_idx, num_classes=num_classes)

    return X_seq, y_cat, y_idx, idx_to_zone

def predict_next_move_using_model(outputdir,mapname,model_path, X_seq, y_idx, idx_to_zone):
    #Load the model
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")

    #Make predictions
    preds_prob = model.predict(X_seq)
    preds_idx = preds_prob.argmax(axis=1)

    #Calculate accuracy
    accuracy = np.mean(preds_idx == y_idx)
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

    # Print a classification report
    y_true_labels = [idx_to_zone[i] for i in y_idx]
    y_pred_labels = [idx_to_zone[i] for i in preds_idx]
    new_replay_rep = classification_report(y_true_labels, y_pred_labels)
    #print("Classification Report:")
    #print(new_replay_rep)
    report_filename = os.path.join(outputdir, f"{mapname}_classification_report.txt")
    with open(report_filename, "w") as f:
        f.write(new_replay_rep)
        f.write(f"Prediction Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report saved")
    # Plot confusion matrix
    plot_confusion_matrix(y_idx, preds_idx, idx_to_zone, f"{mapname}_newreplay", outputdir)

    return accuracy, y_true_labels, y_pred_labels


# main pipeline
#def main():
#you have to give output folder which comes from the main.
#data could be stored externally, pass in the external data folder. If not, i have a local folder with 1 or 2 demos.
def run_nn_pipeline(output_dir, ticks_folder_path=local_ticks_folder_path):
    print("\n\n=== NN pipeline ===")
    nn_op_folder = output_dir
    print(f"The ticks data folder is {ticks_folder_path} and output dir is {nn_op_folder}")
    os.makedirs(nn_op_folder, exist_ok=True)
    df_all = load_and_combine_files(ticks_folder_path)
    maps = df_all["map_name"].unique()
    print(f"Maps found:{maps}")
    for mapname in maps:
        print("**************** ehre are the maps")
        print(f"Processing map: {mapname}")
        df_map = df_all[df_all["map_name"] == mapname].reset_index(drop=True)

        #the accuracy is being affected because most of the time players are not moving
        #since we need to predict the movement, filtering out our rows with no movement.
        #that way tht emodel will learn movement predections

        #df_map = df_map[df_map["current_zone"].astype(str) != df_map["next_zone"].astype(str)]
        #df_map = df_map.reset_index(drop=True)
        #print("removed the non movemenbt trows")
        #this doesnt work either
        #looks like i am removing this before LSTM learns of the sequence of events and the changes.
        #I need to let the model learn this pattern.
        #I think i should filter just before evaluation after seq is done byu lstm

        # Preprocess map
        df_map_clean, X_num_scaled, current_zone, labels, scaler = preprocess_map(df_map)

        if len(df_map_clean) == 0:
            print("No usable rows for map:", mapname)
            continue

        # Build zone index mapping
        zone_to_idx, idx_to_zone = build_zone_index_mapping(current_zone, labels)

        # Create sequences per player timeline
        X_seq, y_idx, change_flags = create_lstm_sequences_per_player(
            df_map_clean, X_num_scaled, current_zone, labels, zone_to_idx, seq_len=SEQ_LEN
        )

        num_classes = len(zone_to_idx)
        y_cat = to_categorical(y_idx, num_classes=num_classes)

        # Train/validation split (simple last-slice split)
        split_idx = int((1 - VALIDATION_SPLIT) * len(X_seq))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_cat[:split_idx], y_cat[split_idx:]
        y_val_idx = y_idx[split_idx:]
        y_train_idx = y_idx[:split_idx]

        print("Train examples:", X_train.shape[0], "Validation examples:", X_val.shape[0])

        # Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, feat_dim)
        model = build_lstm_model(input_shape, num_classes)

        es = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=2
        )

        # Save model
        model_path = os.path.join(nn_op_folder, f"{mapname}_lstm.h5")
        model.save(model_path)
        print(f"Saved LSTM model to:{model_path}")

        # Plot training metrics
        plot_training_history(history, mapname, nn_op_folder)

        # Evaluate on validation set
        preds_prob = model.predict(X_val)
        preds_idx = preds_prob.argmax(axis=1)

        # Filter only rows where zone changed
        change_val = change_flags[split_idx:]
        mask = (change_val == True)

        filtered_true_idx = y_val_idx[mask]
        filtered_pred_idx = preds_idx[mask]

        y_true_labels = [idx_to_zone[i] for i in filtered_true_idx]
        y_pred_labels = [idx_to_zone[i] for i in filtered_pred_idx]

        #print(classification_report(y_true_labels, y_pred_labels))

        # confusion matrix plot
        #plot_confusion_matrix(y_val_idx, preds_idx, idx_to_zone, mapname, OUTPUT_FOLDER)
        #enhancing the cm for showing non staying predictions
        mask = (change_val == True)

        filtered_true = y_val_idx[mask]
        filtered_pred = preds_idx[mask]

        plot_confusion_matrix(filtered_true, filtered_pred, idx_to_zone, mapname, nn_op_folder)

    #I am plannint to use new unseen replays to really test the accuracy

        #dust2_new_replay_path = r"C:\gatech\ticks\newreplay\furia-vs-the-mongolz-m1-dust2_ticks.csv"
        #dust2_new_replay_path = r"C:\gatech\ticks\newreplay\pain-vs-the-mongolz-m1-dust2_ticks.csv"
        #mirage_new_replay_path = r"C:\gatech\ticks\newreplay\the-mongolz-vs-liquid-m2-mirage_ticks.csv"
        dust2_model_path = os.path.join(nn_op_folder, f"dust2_lstm.h5")
        mirage_model_path = os.path.join(nn_op_folder, f"mirage_lstm.h5")
        newreplay_folder_path = fr"{ticks_folder_path}\test"
        if not os.path.exists(newreplay_folder_path):
            print(f"Not testing using new replays as folder not present: {newreplay_folder_path}")  # graceful return
        else:
            candidates = [
                os.path.join(newreplay_folder_path, f)
                for f in os.listdir(newreplay_folder_path)
                if mapname in f and f.endswith("_ticks.csv")
            ]
            if not candidates:
                new_replay_path = None
            new_replay_path = max(candidates, key=os.path.getmtime)

            if(mapname=="dust2" and new_replay_path is not None):
                print(f"using {new_replay_path} to test the model {dust2_model_path}")
                X_seq_new, y_cat_new, y_idx_new, idx_to_zone_new = load_and_preprocess_new_replay(new_replay_path)

                if X_seq_new is not None:
                    # Use the trained model to predict next move on the new replay data
                    accuracy, y_true_labels, y_pred_labels = predict_next_move_using_model(nn_op_folder,
                        mapname,dust2_model_path, X_seq_new, y_idx_new, idx_to_zone_new
                    )
                    print(f"Prediction accuracy on dust2 new replay: {accuracy * 100:.2f}%")


            if (mapname=="mirage" and new_replay_path is not None):
                print(f"using {new_replay_path} to test the model {mirage_model_path}")
                X_seq_new, y_cat_new, y_idx_new, idx_to_zone_new = load_and_preprocess_new_replay(new_replay_path)

                if X_seq_new is not None:
                    # Use the trained model to predict next move on the new replay data
                    accuracy, y_true_labels, y_pred_labels = predict_next_move_using_model(nn_op_folder,
                        mapname,mirage_model_path, X_seq_new, y_idx_new, idx_to_zone_new
                    )
                    print(f"Prediction accuracy on mirage new replay: {accuracy * 100:.2f}%")
    print("\nAll maps processed. Done.")

def main():
    run_nn_pipeline(nn_op_folder, local_ticks_folder_path)

if __name__ == "__main__":
    main()
