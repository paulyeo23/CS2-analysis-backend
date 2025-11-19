import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
import visualkeras
from ann_visualizer.visualize import ann_viz

class WinProbabilityModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def generate_dummy_data(self, n_samples=5000):
        np.random.seed(42)
        data = pd.DataFrame({
            'time_elapsed': np.random.randint(0, 115, n_samples),
            'team_money': np.random.randint(5000, 40000, n_samples),
            'enemy_money': np.random.randint(5000, 40000, n_samples),
            'team_alive': np.random.randint(0, 6, n_samples),
            'enemy_alive': np.random.randint(0, 6, n_samples),
            'bomb_planted': np.random.randint(0, 2, n_samples),
            'bomb_site_A': np.random.randint(0, 2, n_samples),
            'bomb_site_B': np.random.randint(0, 2, n_samples),
            'team_flashes': np.random.randint(0, 6, n_samples),
            'enemy_flashes': np.random.randint(0, 6, n_samples),
            'round_number': np.random.randint(1, 30, n_samples),
            'score_diff': np.random.randint(-10, 11, n_samples),
        })
        data['win'] = (
            0.3*(data['team_alive'] - data['enemy_alive']) +
            0.00001*(data['team_money'] - data['enemy_money']) +
            0.4*data['bomb_planted'] +
            np.random.normal(0, 0.2, n_samples)
        )
        data['win'] = (data['win'] > 0).astype(int)
        return data

    def build_model(self, input_dim):
        
        model = models.Sequential([
            layers.Input(shape=(input_dim,), name="Input_Layer"),
            layers.Dense(24, activation='relu', name="Hidden_1"),
            layers.Dense(36, activation='relu', name="Hidden_2"),
            layers.Dense(1, activation='sigmoid', name="Output_Layer")
        ])
        
        #model = models.Sequential([
        #    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        #    layers.Dense(64, activation='relu'),
        #    layers.Dense(1, activation='sigmoid')
        #])
        #model.compile(optimizer='adam',
        #              loss='binary_crossentropy',
        #              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        self.model = model
        return model

    def train(self, data, epochs=20):
        #X = data
        #print(X)
        #print(X['win'])
       #X=X.drop(columns=['win'])
        #print(X,X.values)
        #the win col values are my y and others are x
        X = data.drop(columns=['win']).values
        y = data['win'].values
        #split train test
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        #print(X_train,y_train)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        #print(X_train)
        #print(X_val)
        #going to build the model
        self.build_model(X_train.shape[1])
        self.model.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=epochs, batch_size=32, verbose=2)
        #loss, acc, auc = self.model.evaluate(X_val, y_val, verbose=0)
        #print(f"Validation Accuracy: {acc:.4f}, AUC: {auc:.4f}")


    def predict(self, state: pd.DataFrame):
        scaled = self.scaler.transform(state)
        return float(self.model.predict(scaled)[0][0])

    def visualize_model(self):
        #plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=True)
        self.model.summary()
        #visualkeras.layered_view(self.model, legend=True).show()      

