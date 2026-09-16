import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import models, layers  
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CNN:
    def __init__(self, train: str, test: str, encoder: LabelEncoder) -> None:
        self.train_dataset = pd.read_pickle(train)
        self.test_dataset = pd.read_pickle(test)
        self.encoder = encoder 

    def run(self):
        # label and value splits
        x_train = np.stack(self.train_dataset["Pixels"].to_numpy())
        x_test = np.stack(self.test_dataset["Pixels"].to_numpy())

        y_train = self.encoder.transform(self.train_dataset["Character"]) 
        y_test = self.encoder.transform(self.test_dataset["Character"])
        num_classes = len(self.encoder.classes_)

        
        cnn = models.Sequential([
            layers.Input(shape=(32, 32, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),  # 0.8 retention probability
            layers.Dense(512, activation="relu"),
            layers.Dense(num_classes, activation="softmax")
        ])

        cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])

        # model fit + prediction
        cnn.fit(x_train, y_train, epochs=50, validation_split=0.1)
        y_pred = np.argmax(cnn.predict(x_test), axis=1)

        # evaluation
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        return accuracy, precision, recall, f1