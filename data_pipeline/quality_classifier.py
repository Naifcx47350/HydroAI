# quality_classifier.py
# -------------------------------
# Water Quality Classification Model
# -------------------------------

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler


class WaterQualityClassifier:
    """
    Classifies water as safe (1) or unsafe (0) based on pH, TDS, turbidity.
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        """
        Expects columns: ['pH', 'TDS', 'turbidity', 'safe']
        Returns scaled features X and labels y.
        """
        X = df[['pH', 'TDS', 'turbidity']].values
        y = df['safe'].values
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def build_model(self, input_dim=3):
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def train(self, X_train, y_train, epochs=20, batch_size=32):
        if self.model is None:
            self.build_model(input_dim=X_train.shape[1])
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop]
        )

    def evaluate(self, X_test, y_test):
        """
        Returns (loss, accuracy).
        """
        return self.model.evaluate(X_test, y_test, verbose=0)

    def predict_quality(self, sample):
        """
        sample shape: (1,3) => [pH, TDS, turbidity].
        Returns predicted class (0 or 1).
        """
        scaled_sample = self.scaler.transform(sample)
        prob = self.model.predict(scaled_sample)
        return (prob > 0.5).astype(int)

    def convert_to_tflite(self, save_path="models/water_quality_classifier.tflite"):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
